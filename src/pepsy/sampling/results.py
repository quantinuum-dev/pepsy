"""Sampling results and fermionic configuration encodings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import autoray as ar
import numpy as np
from ._common import (
    _backend_array_to_numpy,
    _mps_array_backend,
)

__all__ = [
    'FermionConfigurationEncoding',
    'MpsBatchSampleResult',
    'MpsDiagonalEstimate',
    'MpsSampleResult',
    'PEPSSampleResult',
]


def _configs_to_sample_result(
    configs,
    probs,
    *,
    Lx,
    Ly,
    one_d_to_two_d,
    basis=None,
    basis_probability=1.0,
    weights=None,
):
    configs_1d = []
    configs_2d = []
    probs_out = []
    for config, prob in zip(configs, probs):
        config = [int(value) for value in config]
        configs_1d.append(config)
        grid = np.zeros((Ly, Lx), dtype=int)
        for site_1d, spin in enumerate(config):
            x, y = one_d_to_two_d[site_1d]
            grid[y, x] = spin
        configs_2d.append(grid)
        probs_out.append(float(prob))
    return MpsSampleResult(
        configs_1d=configs_1d,
        configs_2d=configs_2d,
        probs=probs_out,
        Lx=Lx,
        Ly=Ly,
        basis=basis,
        basis_probability=basis_probability,
        weights=probs_out if weights is None else weights,
    )


@dataclass
class PEPSSampleResult:
    """Container for PEPS sampling results.

    Attributes
    ----------
    configs
        Sampled physical configurations in the sampler's site order. The
        direct ``PepsSampler`` uses increasing ``y`` then increasing ``x``.
    omegas
        Pair ``(mantissas, exponents)`` for proposal probabilities.
    ps
        Pair ``(mantissas, exponents)`` for sampled PEPS amplitudes.
    log_probabilities, log_abs_amplitudes, log_weights
        Natural-log NumPy arrays computed from the scaled pairs without
        materializing their powers of ten. Access copies any backend scalars
        to the host. ``log_weights`` represents ``log(|Psi|**2 / q)``.
    """

    configs: list[list[int]]
    omegas: tuple[list[float], list[int]]
    ps: tuple[list[Any], list[Any]]

    def __len__(self):
        """Return the number of sampled configurations."""
        return len(self.configs)

    @staticmethod
    def _scaled_logs(pair, *, absolute=False):
        mantissas, exponents = pair
        values = np.asarray([
            _backend_array_to_numpy(value).item() for value in mantissas
        ])
        powers = np.asarray([
            _backend_array_to_numpy(value).item() for value in exponents
        ], dtype=float)
        if absolute:
            values = np.abs(values)
        with np.errstate(divide="ignore"):
            return np.log(values) + powers * math.log(10.0)

    @property
    def log_probabilities(self):
        """Natural logs of the sampled proposal probabilities, as a NumPy array."""
        return self._scaled_logs(self.omegas)

    @property
    def log_abs_amplitudes(self):
        """Natural logs of the absolute PEPS amplitudes, as a NumPy array."""
        return self._scaled_logs(self.ps, absolute=True)

    @property
    def log_weights(self):
        """Natural logs of unnormalized importance weights ``|Psi|**2 / q``."""
        return 2.0 * self.log_abs_amplitudes - self.log_probabilities


@dataclass
class MpsSampleResult:
    """Container for MPS samples with 2D coordinate mapping.

    Attributes
    ----------
    configs_1d : list[list[int]]
        Each entry is a list of length L with spin indices (0 or 1).
    configs_2d : list[np.ndarray]
        Each entry is a (Ly, Lx) int array with spin indices on the 2D lattice.
    probs : list[float]
        Born probability ``|⟨config|ψ⟩|²`` for each sample.
    Lx : int
        Lattice width.
    Ly : int
        Lattice height.
    basis : tuple[str, ...] | None
        Pauli basis used at each site. ``None`` means the legacy sampler did
        not attach basis metadata; ``VecSampler`` always records it.
    basis_probability : float
        Probability of selecting the resolved basis pattern under the basis
        proposal. Fixed bases have probability one; ``VecSampler``'s
        ``basis="random"`` has probability ``3**(-L)``.
    weights : list[float]
        Joint proposal probability of each returned basis/outcome pair. This
        equals ``probs`` for fixed-basis sampling.
    """

    configs_1d: list[list[int]]
    configs_2d: list[np.ndarray]
    probs: list[float]
    Lx: int
    Ly: int
    basis: tuple[str, ...] | None = None
    basis_probability: float = 1.0
    weights: list[float] | None = None

    def __post_init__(self):
        self.basis_probability = float(self.basis_probability)
        if not np.isfinite(self.basis_probability) or self.basis_probability <= 0:
            raise ValueError("basis_probability must be finite and positive")
        if self.weights is None:
            self.weights = self.probs
        if len(self.weights) != len(self.probs):
            raise ValueError("weights and probs must have the same length")

    def __len__(self):
        return len(self.configs_1d)

    def magnetizations(self) -> np.ndarray:
        """Per-sample magnetization ⟨M⟩ = (1/L) Σ (1 - 2·spin_i)."""
        L = self.Lx * self.Ly
        return np.array([
            np.sum(1 - 2 * np.array(c)) / L for c in self.configs_1d
        ])


@dataclass(frozen=True)
class MpsDiagonalEstimate:
    """Monte Carlo estimate of a diagonal MPS observable.

    Attributes
    ----------
    mean
        Sample mean of the observable.
    standard_error
        Standard error estimated from the unbiased sample variance. It is
        ``nan`` when only one sample was requested, because no variance
        estimate is available.
    n_samples
        Number of Born samples used for the estimate.
    observable
        Canonical observable name accepted by
        :meth:`MpsSampler.estimate_fermion_diagonal`.
    sites
        Physical sites averaged or summed by a one-site observable.
    pairs
        Physical pairs averaged by a density-correlation observable.
    """

    mean: float
    standard_error: float
    n_samples: int
    observable: str
    sites: tuple[int, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class FermionConfigurationEncoding:
    """Symmetry-aware meaning of sampled fermionic physical codes.

    ``MpsSampler`` always returns *physical codes*: integer positions in the
    MPS physical legs. They must be decoded before a caller interprets them as
    spinful occupations. For example, a spinful parity ``Z2`` MPS uses the
    physical order ``empty, double, up, down``, whereas its resolved U1 path
    uses ``empty, down, up, double``.

    The encoding is site-aware, immutable, and can convert between physical
    configurations and occupation configurations without relying on either
    convention implicitly. Spinful occupations have shape
    ``(batch, n_sites, 2)`` in ``(n_up, n_down)`` order; spinless occupations
    have shape ``(batch, n_sites)``.
    """

    symmetry: str
    spinful: bool
    code_to_occupations: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self):
        symmetry = str(self.symmetry).upper()
        spinful = bool(self.spinful)
        width = 2 if spinful else 1
        tables = tuple(
            tuple(tuple(int(value) for value in occupation) for occupation in table)
            for table in self.code_to_occupations
        )
        if not tables or any(not table for table in tables):
            raise ValueError("A fermion configuration encoding needs every site map.")
        for table in tables:
            if len(set(table)) != len(table):
                raise ValueError("Each physical code must represent one occupation.")
            if any(
                len(occupation) != width
                or any(value not in {0, 1} for value in occupation)
                for occupation in table
            ):
                raise ValueError(
                    "Fermion occupation entries must contain binary "
                    f"{'(n_up, n_down)' if spinful else 'occupation'} values."
                )
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "spinful", spinful)
        object.__setattr__(self, "code_to_occupations", tables)

    @property
    def n_sites(self) -> int:
        """Number of physical MPS sites covered by this encoding."""
        return len(self.code_to_occupations)

    @property
    def physical_dims(self) -> tuple[int, ...]:
        """Per-site physical-code dimensions."""
        return tuple(len(table) for table in self.code_to_occupations)

    def site_code_map(self, site: int) -> dict[int, tuple[int, ...]]:
        """Return a copy of the physical-code map for one site."""
        site = int(site)
        if not 0 <= site < self.n_sites:
            raise ValueError(f"site must be in 0..{self.n_sites - 1}.")
        return dict(enumerate(self.code_to_occupations[site]))

    def _config_rows(self, physical_configs):
        rows = np.asarray(_backend_array_to_numpy(physical_configs), dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.n_sites:
            raise ValueError(
                "physical_configs must have shape "
                f"(batch, n_sites={self.n_sites}); got {tuple(rows.shape)}."
            )
        for site, dim in enumerate(self.physical_dims):
            invalid = (rows[:, site] < 0) | (rows[:, site] >= dim)
            if np.any(invalid):
                values = np.unique(rows[invalid, site]).tolist()
                raise ValueError(
                    f"physical_configs contain invalid code(s) at site {site}: "
                    f"{values!r}."
                )
        return rows

    def decode(self, physical_configs, *, to_numpy: bool = False):
        """Decode physical codes into on-site occupations.

        The result remains on Torch/CuPy when ``physical_configs`` is on that
        backend, unless ``to_numpy=True`` is requested.
        """
        rows = self._config_rows(physical_configs)
        backend = _mps_array_backend(physical_configs)
        width = 2 if self.spinful else 1

        if to_numpy or backend not in {"torch", "cupy"}:
            out = np.empty(
                rows.shape + ((width,) if self.spinful else ()),
                dtype=np.int64,
            )
            for site, table in enumerate(self.code_to_occupations):
                values = np.asarray(table, dtype=np.int64)
                decoded = values[rows[:, site]]
                if self.spinful:
                    out[:, site, :] = decoded
                else:
                    out[:, site] = decoded[:, 0]
            return out

        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            codes = physical_configs.to(dtype=torch.long)
            shape = tuple(codes.shape) + ((width,) if self.spinful else ())
            out = torch.empty(shape, dtype=torch.long, device=codes.device)
            for site, table in enumerate(self.code_to_occupations):
                values = torch.as_tensor(table, dtype=torch.long, device=codes.device)
                decoded = values[codes[:, site]]
                if self.spinful:
                    out[:, site, :] = decoded
                else:
                    out[:, site] = decoded[:, 0]
            return out

        import cupy as cp  # pylint: disable=import-outside-toplevel

        codes = cp.asarray(physical_configs, dtype=cp.int64)
        shape = tuple(codes.shape) + ((width,) if self.spinful else ())
        out = cp.empty(shape, dtype=cp.int64)
        for site, table in enumerate(self.code_to_occupations):
            values = cp.asarray(table, dtype=cp.int64)
            decoded = values[codes[:, site]]
            if self.spinful:
                out[:, site, :] = decoded
            else:
                out[:, site] = decoded[:, 0]
        return out

    occupations = decode

    def encode(self, occupations, *, to_numpy: bool = False):
        """Encode occupations as the physical codes of the sampled MPS."""
        values = np.asarray(_backend_array_to_numpy(occupations), dtype=np.int64)
        expected_shape = (
            (values.shape[0], self.n_sites, 2)
            if self.spinful and values.ndim >= 1
            else (values.shape[0], self.n_sites)
            if not self.spinful and values.ndim >= 1
            else None
        )
        if expected_shape is None or tuple(values.shape) != expected_shape:
            suffix = ", 2" if self.spinful else ""
            raise ValueError(
                "occupations must have shape "
                f"(batch, n_sites={self.n_sites}{suffix}); got {tuple(values.shape)}."
            )
        if np.any((values < 0) | (values > 1)):
            raise ValueError("occupations must contain only zero and one values.")

        rows = np.empty((values.shape[0], self.n_sites), dtype=np.int64)
        for site, table in enumerate(self.code_to_occupations):
            inverse = {occupation: code for code, occupation in enumerate(table)}
            site_values = values[:, site, :] if self.spinful else values[:, site, None]
            for row, occupation in enumerate(site_values):
                try:
                    rows[row, site] = inverse[tuple(int(value) for value in occupation)]
                except KeyError as exc:
                    raise ValueError(
                        f"occupation {tuple(occupation)!r} is unavailable at site {site}."
                    ) from exc

        backend = _mps_array_backend(occupations)
        if to_numpy or backend not in {"torch", "cupy"}:
            return rows
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            return torch.as_tensor(rows, dtype=torch.long, device=occupations.device)
        import cupy as cp  # pylint: disable=import-outside-toplevel

        return cp.asarray(rows, dtype=cp.int64)


@dataclass
class MpsBatchSampleResult:
    """Backend-native batched MPS samples.

    Attributes
    ----------
    configs
        Array-like object with shape ``(n_samples, L)``. With the native
        sampler this can be a NumPy array, Torch tensor, or CuPy array.
        Symmray MPS samples use the backend of their underlying blocks.
    probs
        Born probabilities for ``configs`` with shape ``(n_samples,)``.
    Lx, Ly
        2D lattice dimensions used by :meth:`configs_2d` and
        :meth:`to_sample_result`.
    one_d_to_two_d
        Mapping from 1D site index to ``(x, y)`` coordinate.
    backend
        Backend of ``configs`` and ``probs``: ``"numpy"``, ``"torch"``, or
        ``"cupy"``.
    basis
        Pauli basis used at each site, when the sampler supplies it.
    basis_probability
        Probability of selecting the resolved basis pattern.
    weights
        Joint proposal probability of each returned basis/outcome pair.
    """

    configs: Any
    probs: Any
    Lx: int
    Ly: int
    one_d_to_two_d: dict[int, tuple[int, int]]
    backend: str = "numpy"
    configuration_encoding: FermionConfigurationEncoding | None = None
    basis: tuple[str, ...] | None = None
    basis_probability: float = 1.0
    weights: Any | None = None

    def __post_init__(self):
        self.basis_probability = float(self.basis_probability)
        if not np.isfinite(self.basis_probability) or self.basis_probability <= 0:
            raise ValueError("basis_probability must be finite and positive")
        if self.weights is None:
            self.weights = self.probs
        if getattr(self.weights, "shape", None) != getattr(self.probs, "shape", None):
            raise ValueError("weights and probs must have the same shape")

    def __len__(self):
        return int(self.configs.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of sampled configurations."""
        return len(self)

    @property
    def L(self) -> int:
        """Number of MPS sites."""
        return int(self.configs.shape[1])

    def to_numpy(self) -> "MpsBatchSampleResult":
        """Return a CPU NumPy copy of this batched result."""
        return MpsBatchSampleResult(
            configs=_backend_array_to_numpy(self.configs),
            probs=_backend_array_to_numpy(self.probs),
            Lx=self.Lx,
            Ly=self.Ly,
            one_d_to_two_d=dict(self.one_d_to_two_d),
            backend="numpy",
            configuration_encoding=self.configuration_encoding,
            basis=self.basis,
            basis_probability=self.basis_probability,
            weights=_backend_array_to_numpy(self.weights),
        )

    def configs_1d(self) -> list[list[int]]:
        """Return configurations as Python ``list[list[int]]``."""
        configs = _backend_array_to_numpy(self.configs)
        return [[int(value) for value in config] for config in configs]

    def configs_2d(self) -> list[np.ndarray]:
        """Return configurations as ``(Ly, Lx)`` NumPy grids."""
        return self.to_sample_result().configs_2d

    def occupations(self, *, to_numpy: bool = False):
        """Decode fermionic physical codes with the attached configuration map."""
        if self.configuration_encoding is None:
            raise ValueError(
                "This batch has no fermion configuration encoding. Pass "
                "fermion=... to MpsSampler.sample_batch(...)."
            )
        return self.configuration_encoding.decode(self.configs, to_numpy=to_numpy)

    def magnetizations(self, *, to_numpy: bool = False):
        """Per-sample magnetization ``(1 / L) * sum_i (1 - 2 * spin_i)``."""
        backend = _mps_array_backend(self.configs)
        if backend == "torch":
            configs = self.configs.to(dtype=self.probs.dtype)
            out = (1 - 2 * configs).sum(dim=1) / float(self.L)
            return ar.to_numpy(out) if to_numpy else out
        if backend == "cupy":
            configs = self.configs.astype(np.float64, copy=False)
            out = (1 - 2 * configs).sum(axis=1) / float(self.L)
            return ar.to_numpy(out) if to_numpy else out
        configs = np.asarray(self.configs, dtype=float)
        return (1 - 2 * configs).sum(axis=1) / float(self.L)

    def to_sample_result(self) -> MpsSampleResult:
        """Convert to the legacy list/grid :class:`MpsSampleResult`."""
        batch = self.to_numpy()
        return _configs_to_sample_result(
            batch.configs,
            batch.probs,
            Lx=batch.Lx,
            Ly=batch.Ly,
            one_d_to_two_d=batch.one_d_to_two_d,
            basis=batch.basis,
            basis_probability=batch.basis_probability,
            weights=batch.weights,
        )
