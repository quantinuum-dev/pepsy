"""Backend-neutral data contracts for PEPS variational Monte Carlo.

The Torch and NetKet integrations intentionally keep different numerical
engines, but they can share the objects that describe an operator problem and
the results returned by sampling, measurement, and optimization.  This module
does not require Torch, JAX, Flax, or NetKet. NumPy is used only for small,
backend-neutral result conveniences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "BackendCapabilityWarning",
    "VMCBackendCapabilityError",
    "ContractionFallbackWarning",
    "ContractionConfig",
    "CompiledOperatorSum",
    "LocalMatrixTerm",
    "MCState",
    "NumericalStabilityWarning",
    "OperatorFactor",
    "OperatorSum",
    "OptimizationConfig",
    "ProductTerm",
    "SamplingConfig",
    "SamplingDiagnosticWarning",
    "SymmetryFallbackWarning",
    "VMCMeasurement",
    "VMCOptimizationResult",
    "VMC",
    "VMCProblem",
    "VMCSamples",
    "VMCWarning",
    "normalize_operator_sum",
]


class VMCWarning(UserWarning):
    """Base warning for non-fatal VMC capability or numerical diagnostics."""


class BackendCapabilityWarning(VMCWarning):
    """A backend cannot use a requested optional acceleration."""


class VMCBackendCapabilityError(NotImplementedError):
    """A requested portable VMC operation is unavailable on one backend.

    The high-level adapters raise this instead of accepting a shared setting
    and silently changing its meaning. Native backend APIs can still expose
    their additional capabilities directly.
    """


class SymmetryFallbackWarning(VMCWarning):
    """A symmetry-aware path selected a slower or less specialized fallback."""


class ContractionFallbackWarning(VMCWarning):
    """A requested contraction optimizer or route required a fallback."""


class SamplingDiagnosticWarning(VMCWarning):
    """Sampling diagnostics indicate insufficient mixing or support."""


class NumericalStabilityWarning(VMCWarning):
    """Amplitudes, logarithms, or update diagnostics approach instability."""


_CONTRACTION_ALIASES = {
    "exact": "exact",
    "hotrg": "hotrg",
    "ctmrg": "ctmrg",
    "boundary": "boundary",
    "mps": "boundary",
    "boundary-mps": "boundary",
    "contract-boundary": "boundary",
}


def _positive_int(name, value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive integer{suffix}.")
    return int(value)


@dataclass(frozen=True)
class ContractionConfig:
    """Shared contraction settings consumed by Torch and NetKet adapters."""

    method: str = "exact"
    chi: int | None = None
    cutoff: float = 0.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        key = str(self.method).replace("_", "-").lower()
        try:
            method = _CONTRACTION_ALIASES[key]
        except KeyError as exc:
            raise ValueError(
                "contraction method must be 'exact', 'hotrg', 'ctmrg', "
                "or 'boundary'."
            ) from exc
        chi = _positive_int("chi", self.chi, allow_none=True)
        if method != "exact" and chi is None:
            raise ValueError(f"chi is required for contraction={method!r}.")
        cutoff = float(self.cutoff)
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


def _resolve_contraction_config(contraction, chi=None, cutoff=None, options=None):
    """Return legacy constructor values for a common contraction config."""
    if not isinstance(contraction, ContractionConfig):
        return contraction, chi, cutoff, options
    if chi is not None and contraction.chi is not None and chi != contraction.chi:
        raise ValueError(
            f"chi={chi} conflicts with ContractionConfig.chi={contraction.chi}."
        )
    if cutoff is not None and float(cutoff) != contraction.cutoff:
        raise ValueError(
            f"cutoff={cutoff} conflicts with ContractionConfig.cutoff="
            f"{contraction.cutoff}."
        )
    if options is not None and dict(options) != dict(contraction.options):
        raise ValueError("contraction options conflict with ContractionConfig.options.")
    return (
        contraction.method,
        contraction.chi if chi is None else chi,
        contraction.cutoff if cutoff is None else cutoff,
        dict(contraction.options) if options is None else options,
    )


@dataclass(frozen=True)
class SamplingConfig:
    """Shared chain-preserving sampling settings.

    ``burn_in`` is the number of discarded *thinning intervals* per chain.
    Thus the native Torch sampler advances each chain
    ``(burn_in + n_samples_per_chain) * thin`` Metropolis sweeps: it discards
    ``burn_in * thin`` sweeps, then retains one configuration after every
    ``thin`` further sweeps.  The returned batch has shape
    ``(n_samples_per_chain, n_chains, n_sites)``.
    """

    n_samples_per_chain: int = 128
    n_chains: int = 16
    burn_in: int = 0
    thin: int = 1
    seed: int | None = None
    sampler_seed: int | None = None
    chunk_size: int | None = None
    proposal: str | None = None

    def __post_init__(self):
        n_samples = _positive_int("n_samples_per_chain", self.n_samples_per_chain)
        n_chains = _positive_int("n_chains", self.n_chains)
        if isinstance(self.burn_in, bool) or not isinstance(self.burn_in, int) or self.burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer.")
        thin = _positive_int("thin", self.thin)
        chunk_size = _positive_int("chunk_size", self.chunk_size, allow_none=True)
        if self.seed is not None and isinstance(self.seed, bool):
            raise ValueError("seed must be an integer or None.")
        if self.sampler_seed is not None and isinstance(self.sampler_seed, bool):
            raise ValueError("sampler_seed must be an integer or None.")
        object.__setattr__(self, "n_samples_per_chain", n_samples)
        object.__setattr__(self, "n_chains", n_chains)
        object.__setattr__(self, "burn_in", int(self.burn_in))
        object.__setattr__(self, "thin", thin)
        object.__setattr__(self, "chunk_size", chunk_size)

    @property
    def n_samples(self):
        """Total requested samples across all chains."""
        return self.n_samples_per_chain * self.n_chains

    def torch_kwargs(self):
        """Return the canonical keyword mapping for Torch samplers.

        The Torch sampler accepts a total ``n_samples`` value, so this
        conversion deliberately multiplies by ``n_chains``.  Keeping that
        conversion here prevents the two backends from silently interpreting
        the same user setting differently.
        """
        if self.seed is not None and self.sampler_seed is not None:
            raise ValueError("Pass either seed or sampler_seed, not both.")
        return {
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "n_discard_per_chain": self.burn_in,
            "n_thin": self.thin,
            "seed": self.seed,
            "sampler_seed": self.sampler_seed,
        }

    def netket_kwargs(self):
        """Return the settings understood by ``MCState.sample``.

        NetKet's sampler owns the chain count, therefore ``n_chains`` is
        retained in this mapping for validation by the setup façade rather
        than passed to ``MCState.sample``.
        """
        return {
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "n_discard_per_chain": self.burn_in,
        }


@dataclass(frozen=True)
class OptimizationConfig:
    """Shared energy-optimization settings."""

    n_steps: int = 1
    method: str = "sr"
    learning_rate: float = 1.0e-3
    diag_shift: float = 1.0e-2
    sr_mode: str = "real"
    energy_shift: float = 0.0
    per_site: int | None = None
    progress: bool = False
    warmup: bool = True

    def __post_init__(self):
        n_steps = _positive_int("n_steps", self.n_steps)
        method = str(self.method).replace("-", "_").lower()
        aliases = {
            "sgd": "sgd",
            "vmc": "sgd",
            "sr": "sr",
            "minsr": "minsr",
            "min_sr": "minsr",
        }
        if method not in aliases:
            raise ValueError("method must be 'sgd', 'sr', or 'minsr'.")
        sr_mode = str(self.sr_mode).replace("_", "-").lower()
        sr_mode_aliases = {
            "real": "real",
            "complex": "complex",
            "holomorphic": "holomorphic",
            "holomorphic-complex": "holomorphic",
            "real-imag": "real-imag",
            "real-imaginary": "real-imag",
        }
        if sr_mode not in sr_mode_aliases:
            raise ValueError(
                "sr_mode must be 'real', 'complex', 'holomorphic', or "
                "'real-imag'."
            )
        learning_rate = float(self.learning_rate)
        diag_shift = float(self.diag_shift)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if diag_shift < 0:
            raise ValueError("diag_shift must be non-negative.")
        per_site = _positive_int("per_site", self.per_site, allow_none=True)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "method", aliases[method])
        object.__setattr__(self, "sr_mode", sr_mode_aliases[sr_mode])
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "diag_shift", diag_shift)
        object.__setattr__(self, "per_site", per_site)


@dataclass(frozen=True)
class CompiledOperatorSum:
    """Backend adapter output containing terms plus an identity constant."""

    backend: str
    terms: Any
    constant: Any = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("CompiledOperatorSum.backend must be non-empty.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class OperatorFactor:
    """One local operator factor in a product term.

    ``name`` is deliberately a string rather than a backend operator object.
    Common names include ``"x"``, ``"number"``, and ``"fermion"``.  Backend
    adapters interpret the name and may use ``spin``/``dagger`` to construct
    their native operator.  Keeping the factor symbolic is what lets the same
    Hamiltonian feed both Torch and NetKet.
    """

    site: Any
    name: str
    spin: str | None = None
    dagger: bool | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("OperatorFactor.name must be a non-empty string.")
        if self.spin is not None and not isinstance(self.spin, str):
            raise TypeError("OperatorFactor.spin must be a string or None.")
        if self.dagger is not None and not isinstance(self.dagger, bool):
            raise TypeError("OperatorFactor.dagger must be a bool or None.")


@dataclass(frozen=True)
class ProductTerm:
    """A coefficient times a product of symbolic local operator factors."""

    coefficient: Any
    factors: tuple[OperatorFactor, ...]

    def __post_init__(self):
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("ProductTerm requires at least one operator factor.")
        if not all(isinstance(factor, OperatorFactor) for factor in factors):
            raise TypeError("ProductTerm.factors must contain OperatorFactor objects.")
        object.__setattr__(self, "factors", factors)

    @property
    def support(self):
        """Sites touched by the term, retaining factor order."""
        return tuple(factor.site for factor in self.factors)


@dataclass(frozen=True)
class LocalMatrixTerm:
    """A coefficient times a local operator on an explicit site support.

    ``matrix`` uses output axes followed by input axes. Thus a one-site
    operator has shape ``(d, d)`` and a two-site operator has shape
    ``(d0, d1, d0, d1)``. This is the convention used by the Torch connection
    kernel; adapters flatten the two axis groups only where their native
    operator API requires a matrix.
    """

    support: tuple[Any, ...]
    matrix: Any
    coefficient: Any = 1.0
    basis: tuple[Any, ...] | None = None

    def __post_init__(self):
        support = tuple(self.support)
        if not support:
            raise ValueError("LocalMatrixTerm.support cannot be empty.")
        if len(set(support)) != len(support):
            raise ValueError("LocalMatrixTerm.support must contain unique sites.")
        shape = getattr(self.matrix, "shape", None)
        if shape is not None:
            shape = tuple(int(size) for size in shape)
            n_sites = len(support)
            if len(shape) != 2 * n_sites:
                raise ValueError(
                    "LocalMatrixTerm.matrix must have output axes followed by "
                    f"input axes: expected rank {2 * n_sites} for support "
                    f"{support!r}, got shape {shape!r}."
                )
            if any(size <= 0 for size in shape):
                raise ValueError("LocalMatrixTerm.matrix dimensions must be positive.")
            if shape[:n_sites] != shape[n_sites:]:
                raise ValueError(
                    "LocalMatrixTerm.matrix output and input dimensions must match."
                )
        basis = None if self.basis is None else tuple(self.basis)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "basis", basis)


@dataclass(frozen=True)
class OperatorSum:
    """Backend-neutral finite sum of local or symbolic operator terms."""

    terms: tuple[ProductTerm | LocalMatrixTerm, ...] = ()
    constant: Any = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        terms = tuple(self.terms)
        if not all(isinstance(term, (ProductTerm, LocalMatrixTerm)) for term in terms):
            raise TypeError(
                "OperatorSum.terms must contain ProductTerm or LocalMatrixTerm objects."
            )
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_terms(cls, terms, *, constant=0.0, metadata=None):
        """Construct an operator sum while normalizing a list or generator."""
        return cls(
            terms=tuple(terms),
            constant=constant,
            metadata={} if metadata is None else metadata,
        )

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    @property
    def sites(self):
        """Sorted-by-first-use tuple of sites appearing in the sum."""
        result = []
        seen = set()
        for term in self.terms:
            support = term.support
            for site in support:
                try:
                    key = site
                    already_seen = key in seen
                except TypeError:
                    key = repr(site)
                    already_seen = key in seen
                if not already_seen:
                    seen.add(key)
                    result.append(site)
        return tuple(result)


def normalize_operator_sum(value, *, constant=0.0, metadata=None):
    """Normalize common legacy term containers to :class:`OperatorSum`.

    A mapping or iterable of ``(support, operator)`` pairs is interpreted as
    the legacy local-matrix form used by Torch VMC.  A sequence of
    :class:`ProductTerm` and :class:`LocalMatrixTerm` objects is preserved.
    Backend-native NetKet operators and other opaque objects are intentionally
    rejected here; callers should pass those through their backend adapter.
    """
    if isinstance(value, OperatorSum):
        if constant == 0.0 and metadata is None:
            return value
        merged_metadata = dict(value.metadata)
        if metadata is not None:
            merged_metadata.update(metadata)
        return OperatorSum(
            terms=value.terms,
            constant=value.constant + constant,
            metadata=merged_metadata,
        )

    if hasattr(value, "terms") and not isinstance(value, (tuple, list, dict)):
        value = value.terms
    if hasattr(value, "items"):
        entries = tuple(value.items())
    else:
        try:
            entries = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "terms must be an OperatorSum, a sequence of common term "
                "objects, or a mapping/iterable of (support, operator) pairs."
            ) from exc

    if all(isinstance(term, (ProductTerm, LocalMatrixTerm)) for term in entries):
        return OperatorSum(
            terms=entries,
            constant=constant,
            metadata={} if metadata is None else metadata,
        )

    matrix_terms = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError(
                "legacy terms must contain (support, operator) pairs or common "
                "OperatorTerm objects."
            )
        support, operator = entry
        if isinstance(support, (tuple, list)):
            normalized_support = tuple(support)
        else:
            normalized_support = (support,)
        matrix_terms.append(
            LocalMatrixTerm(support=normalized_support, matrix=operator)
        )
    return OperatorSum(
        terms=tuple(matrix_terms),
        constant=constant,
        metadata={} if metadata is None else metadata,
    )


def _canonical_fermion_name(factor):
    """Map a symbolic factor to Pepsy's local fermion operator names."""
    name = str(factor.name).lower()
    aliases = {
        "n": "number",
        "occupation": "number",
        "number_up": "number_u",
        "n_up": "number_u",
        "number_down": "number_d",
        "n_down": "number_d",
        "create_up": "create_u",
        "create_down": "create_d",
        "annihilate_up": "annihilate_u",
        "annihilate_down": "annihilate_d",
        "destroy": "annihilate",
        "destroy_up": "annihilate_u",
        "destroy_down": "annihilate_d",
        "doublon": "double",
    }
    name = aliases.get(name, name)
    if name == "fermion":
        if factor.dagger is None:
            raise ValueError("fermion factors require dagger=True or False.")
        action = "create" if factor.dagger else "annihilate"
        if factor.spin is None:
            return action
        spin = str(factor.spin).lower()
        if spin in {"up", "u", "↑", "+"}:
            return f"{action}_u"
        if spin in {"down", "d", "↓", "-"}:
            return f"{action}_d"
        raise ValueError("fermion factor spin must be 'up', 'down', or None.")
    if name in {"create", "annihilate", "number"} and factor.spin is not None:
        spin = str(factor.spin).lower()
        suffix = "u" if spin in {"up", "u", "↑", "+"} else (
            "d" if spin in {"down", "d", "↓", "-"} else None
        )
        if suffix is None:
            raise ValueError("fermion factor spin must be 'up' or 'down'.")
        return f"{name}_{suffix}"
    return name


def _expand_fermion_factor(factor):
    """Expand number/double aliases into creation/annihilation factors."""
    name = _canonical_fermion_name(factor)
    site = factor.site
    expansions = {
        "number": ("create", "annihilate"),
        "number_u": ("create_u", "annihilate_u"),
        "number_d": ("create_d", "annihilate_d"),
        "double": (
            "create_u",
            "annihilate_u",
            "create_d",
            "annihilate_d",
        ),
    }
    return tuple((site, item) for item in expansions.get(name, (name,)))


@dataclass(frozen=True)
class VMCProblem:
    """Compatibility bundle for the earlier portable VMC construction API.

    ``hamiltonian`` and entries in ``observables`` may be an :class:`OperatorSum`
    or a backend-native object during the compatibility transition.  New code
    should prefer :class:`MCState` plus :class:`VMC`, while still using
    ``OperatorSum`` so Torch and NetKet receive identical terms.
    """

    peps: Any
    hamiltonian: Any
    observables: Mapping[str, Any] = field(default_factory=dict)
    symmetry: str | None = None
    site_order: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        observables = dict(self.observables)
        if any(not isinstance(name, str) or not name for name in observables):
            raise ValueError("VMCProblem observable names must be non-empty strings.")
        site_order = None if self.site_order is None else tuple(self.site_order)
        if site_order is not None and len(set(site_order)) != len(site_order):
            raise ValueError("VMCProblem.site_order must contain unique sites.")
        object.__setattr__(self, "observables", MappingProxyType(observables))
        object.__setattr__(self, "site_order", site_order)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, init=False)
class MCState:
    """Portable, NetKet-shaped Monte Carlo variational-state recipe.

    ``MCState`` owns the variational ansatz and the sampling specification;
    :class:`VMC` attaches a Hamiltonian and builds the selected backend. The
    familiar ``n_samples`` value is the total over all chains, exactly as in
    NetKet. ``sampling=SamplingConfig(...)`` remains available when the
    per-chain convention is more convenient.
    """

    peps: Any
    sampling: SamplingConfig
    symmetry: str | None
    site_order: tuple[Any, ...] | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        peps,
        *,
        sampling=None,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        thin=None,
        seed=None,
        sampler_seed=None,
        chunk_size=None,
        proposal=None,
        symmetry=None,
        site_order=None,
        metadata=None,
    ):
        if sampling is not None:
            if not isinstance(sampling, SamplingConfig):
                raise TypeError("sampling must be a SamplingConfig or None.")
            supplied = {
                "n_samples": n_samples,
                "n_chains": n_chains,
                "n_discard_per_chain": n_discard_per_chain,
                "thin": thin,
                "seed": seed,
                "sampler_seed": sampler_seed,
                "chunk_size": chunk_size,
                "proposal": proposal,
            }
            conflicting = [name for name, value in supplied.items() if value is not None]
            if conflicting:
                raise ValueError(
                    "Pass either sampling=... or direct MCState sampling "
                    f"options, not both; got {', '.join(conflicting)}."
                )
        else:
            n_chains = 16 if n_chains is None else _positive_int("n_chains", n_chains)
            n_samples = 1024 if n_samples is None else _positive_int(
                "n_samples", n_samples
            )
            if n_samples % n_chains:
                raise ValueError(
                    "n_samples must be divisible by n_chains so every chain "
                    "has the same retained length."
                )
            sampling = SamplingConfig(
                n_samples_per_chain=n_samples // n_chains,
                n_chains=n_chains,
                burn_in=0 if n_discard_per_chain is None else n_discard_per_chain,
                thin=1 if thin is None else thin,
                seed=seed,
                sampler_seed=sampler_seed,
                chunk_size=chunk_size,
                proposal=proposal,
            )

        site_order = None if site_order is None else tuple(site_order)
        if site_order is not None and len(set(site_order)) != len(site_order):
            raise ValueError("MCState.site_order must contain unique sites.")
        object.__setattr__(self, "peps", peps)
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "site_order", site_order)
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))

    @property
    def ansatz(self):
        """Alias for the variational tensor-network ansatz."""
        return self.peps

    @property
    def model(self):
        """NetKet-style alias for the variational ansatz."""
        return self.peps

    @property
    def n_samples(self):
        """Total retained samples over all chains."""
        return self.sampling.n_samples

    @property
    def n_chains(self):
        """Number of independent Markov chains requested for this state."""
        return self.sampling.n_chains

    @property
    def n_discard_per_chain(self):
        """Per-chain burn-in, using NetKet's naming convention."""
        return self.sampling.burn_in

    def to_problem(self, hamiltonian, *, observables=None):
        """Make the compatibility :class:`VMCProblem` representation."""
        return VMCProblem(
            peps=self.peps,
            hamiltonian=hamiltonian,
            observables={} if observables is None else observables,
            symmetry=self.symmetry,
            site_order=self.site_order,
            metadata=self.metadata,
        )


class VMC:
    """Portable NetKet-style VMC driver over a Pepsy ``MCState``.

    The class owns a Hamiltonian and a variational state recipe, then exposes
    ``sample()``, ``expect()``, and ``run()``. Backend-specific state and
    driver controls remain reachable through :attr:`native`.
    """

    def __init__(
        self,
        hamiltonian,
        variational_state,
        *,
        backend="torch",
        fermion=None,
        observables=None,
        contraction=None,
        **build_kwargs,
    ):
        if not isinstance(variational_state, MCState):
            raise TypeError("variational_state must be an MCState.")
        backend = str(backend).lower()
        if backend not in {"torch", "netket"}:
            raise ValueError("backend must be 'torch' or 'netket'.")

        self.hamiltonian = hamiltonian
        self.variational_state = variational_state
        self.backend = backend
        self.problem = variational_state.to_problem(
            hamiltonian,
            observables=observables,
        )
        if backend == "torch":
            from .torch import build_torch_vmc

            self._setup = build_torch_vmc(
                self.problem,
                fermion=fermion,
                contraction=contraction,
                sampling=variational_state.sampling,
                **build_kwargs,
            )
        else:
            from .netket import build_netket_vmc

            self._setup = build_netket_vmc(
                self.problem,
                fermion=fermion,
                contraction=contraction,
                sampling=variational_state.sampling,
                **build_kwargs,
            )

    @property
    def state(self):
        """Alias for :attr:`variational_state`."""
        return self.variational_state

    @property
    def native(self):
        """The underlying native Torch or NetKet setup."""
        return self._setup.native

    @property
    def setup(self):
        """Portable backend setup used by this driver."""
        return self._setup

    def sample(self, sampling=None):
        """Collect samples through the selected backend."""
        return self._setup.sample(sampling)

    def measure(
        self,
        observables=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """Measure energy and optional observables.

        ``samples`` optionally supplies a previously collected or external
        batch. Torch accepts fixed ``weights`` or ``proposal_log_probs`` for
        self-normalized importance sampling; the latter is preferred when the
        batch will also be reused for optimization.
        """
        kwargs = {"sampling": sampling}
        if samples is not None:
            kwargs["samples"] = samples
        if weights is not None:
            kwargs["weights"] = weights
        if proposal_log_probs is not None:
            kwargs["proposal_log_probs"] = proposal_log_probs
        return self._setup.measure(observables, **kwargs)

    def expect(
        self,
        observable=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """NetKet-style expectation call returning a common measurement.

        With no argument this measures the Hamiltonian. Supplying one
        observable adds it under the ``"expectation"`` key alongside energy.
        """
        if observable is None:
            return self.measure(
                sampling=sampling,
                samples=samples,
                weights=weights,
                proposal_log_probs=proposal_log_probs,
            )
        return self.measure(
            {"expectation": observable},
            sampling=sampling,
            samples=samples,
            weights=weights,
            proposal_log_probs=proposal_log_probs,
        )

    def check_mc_convergence(self, **kwargs):
        """Run the selected backend's explicit post-run mixing diagnostic."""
        return self._setup.check_mc_convergence(**kwargs)

    def optimize(self, optimization=None, *, n_steps=None, **kwargs):
        """Optimize the variational state through the selected backend."""
        return self._setup.optimize(
            optimization=optimization,
            n_steps=n_steps,
            **kwargs,
        )

    def run(self, n_iter=None, *, optimization=None, **kwargs):
        """NetKet-style optimization entry point returning the run history."""
        if optimization is not None:
            if n_iter is not None and n_iter != optimization.n_steps:
                raise ValueError("n_iter conflicts with optimization.n_steps.")
            return self.optimize(optimization=optimization, **kwargs)
        if n_iter is None:
            raise TypeError("n_iter is required unless optimization is supplied.")
        return self.optimize(n_steps=n_iter, **kwargs)


@dataclass(frozen=True)
class VMCSamples:
    """Backend-neutral Monte Carlo or externally supplied samples.

    ``weights`` defines a fixed weighted empirical batch. Alternatively,
    ``proposal_log_probs`` records ``log q(x)`` so a Torch importance estimate
    can recompute ``|psi_theta(x)|**2 / q(x)`` as parameters change.
    """

    configs: Any
    amplitudes: Any | None = None
    log_amplitudes: Any | None = None
    n_samples_per_chain: int | None = None
    n_chains: int | None = None
    acceptance_rate: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    native: Any | None = None
    weights: Any | None = None
    proposal_log_probs: Any | None = None

    def __post_init__(self):
        shape = getattr(self.configs, "shape", None)
        if shape is not None and len(shape) not in (2, 3):
            raise ValueError(
                "VMCSamples.configs must have shape (samples, sites) or "
                "(samples_per_chain, chains, sites)."
            )
        if self.n_samples_per_chain is not None and self.n_samples_per_chain <= 0:
            raise ValueError("n_samples_per_chain must be positive when supplied.")
        if self.n_chains is not None and self.n_chains <= 0:
            raise ValueError("n_chains must be positive when supplied.")
        if self.weights is not None and self.proposal_log_probs is not None:
            raise ValueError("Pass either weights or proposal_log_probs, not both.")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def chain_shape(self):
        """Return the retained ``(samples_per_chain, chains)`` shape."""
        shape = getattr(self.configs, "shape", None)
        if shape is not None and len(shape) == 3:
            return tuple(int(value) for value in shape[:2])
        if self.n_samples_per_chain is not None or self.n_chains is not None:
            return self.n_samples_per_chain, self.n_chains
        return None


@dataclass(frozen=True)
class VMCMeasurement:
    """Backend-neutral observable or local-energy measurement result."""

    energy_mean: Any | None = None
    energy_variance: Any | None = None
    energy_stderr: Any | None = None
    observables: Mapping[str, Any] = field(default_factory=dict)
    local_values: Any | None = None
    effective_sample_size: Any | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    native: Any | None = None

    def __post_init__(self):
        object.__setattr__(self, "observables", MappingProxyType(dict(self.observables)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def energy(self):
        """Return the backend-neutral sampled energy mean."""
        return self.energy_mean

    @property
    def native_energy(self):
        """Return the backend-specific energy estimate, when retained."""
        return self.observables.get("energy")


@dataclass(frozen=True)
class VMCOptimizationResult:
    """Common optimization history for Torch and NetKet VMC runs."""

    steps: Any
    energies: Any
    errors: Any
    variances: Any | None = None
    final_energy: float | None = None
    final_error: float | None = None
    energy_shift: float = 0.0
    per_site: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    native: Any | None = None

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        per_site = _positive_int("per_site", self.per_site, allow_none=True)
        object.__setattr__(self, "per_site", per_site)
        if self.final_energy is None:
            values = np.asarray(self.energies)
            if values.size:
                object.__setattr__(self, "final_energy", float(np.real(values[-1])))
        if self.final_error is None:
            values = np.asarray(self.errors)
            if values.size:
                object.__setattr__(self, "final_error", float(np.real(values[-1])))

    @property
    def shifted_energies(self):
        """Return energies with the display-only energy shift applied."""
        return np.asarray(self.energies, dtype=float) + float(self.energy_shift)

    @property
    def displayed_energies(self):
        """Return shifted energies, divided by ``per_site`` when requested."""
        values = self.shifted_energies
        return values if self.per_site is None else values / self.per_site
