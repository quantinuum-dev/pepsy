"""Samplers for MPS and PEPS tensor networks."""

from __future__ import annotations

from pepsy._internal.cutoff import dtype_auto_cutoff
from pepsy._internal.quimb import call_quimb_2d
from contextlib import nullcontext
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Iterable
import warnings

import autoray as ar
import numpy as np
from tqdm import tqdm

sample_d2bp = None
build_optimizer = None

__all__ = [
    "FermionConfigurationEncoding",
    "MpsDiagonalEstimate",
    "MpsBatchSampleResult",
    "MpsSampleResult",
    "MpsSampler",
    "PEPSSampleResult",
    "PepsSampler",
    "PepsBpSampler",
    "VecSampler",
]


# CUDA torch.multinomial rejects categorical distributions above this size.
# Keep the limit explicit so the large-state fallback is easy to test.
_TORCH_MULTINOMIAL_MAX_CATEGORIES = 1 << 24


def _validate_one_d_to_two_d(
    one_d_to_two_d: dict[int, tuple[int, int]],
    *,
    expected_L: int | None = None,
) -> int:
    """Validate a complete 0..L-1 site map and return ``L``."""
    if not one_d_to_two_d:
        raise ValueError("one_d_to_two_d must contain at least one site.")
    L = int(expected_L) if expected_L is not None else len(one_d_to_two_d)
    if L < 1:
        raise ValueError("expected_L must be >= 1.")
    expected = set(range(L))
    got = set(one_d_to_two_d)
    if got != expected:
        raise ValueError(
            "one_d_to_two_d keys must be exactly consecutive site indices "
            f"0..{L - 1}; got {sorted(got)!r}."
        )
    for site, coord in one_d_to_two_d.items():
        if not (
            isinstance(coord, tuple)
            and len(coord) == 2
            and all(isinstance(value, int) for value in coord)
        ):
            raise TypeError(
                "one_d_to_two_d values must be (x, y) integer tuples; "
                f"site {site!r} maps to {coord!r}."
            )
    return L


def _normalize_mps_sampler_backend(backend):
    if backend is None:
        return "quimb"
    key = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "quimb": "quimb",
        "cpu": "quimb",
        "numpy_quimb": "quimb",
        "auto": "auto",
        "native": "native",
        "device": "native",
        "cuda": "native",
        "gpu": "native",
        "numpy": "numpy",
        "np": "numpy",
        "torch": "torch",
        "pytorch": "torch",
        "cupy": "cupy",
        "cp": "cupy",
        "symmray": "symmray",
        "symmetric": "symmray",
        "block_sparse": "symmray",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown MpsSampler backend {backend!r}. Expected one of: {allowed}."
        ) from exc


_MEASUREMENT_BASIS_LABELS = frozenset(("X", "Y", "Z"))
_DEFAULT_VEC_SAMPLE_CHUNK_SIZE = 4096


def _validate_sample_count(n_samples):
    """Normalize a non-negative integer shot count."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError(f"n_samples must be an integer, got {n_samples!r}")
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    return n_samples


def _validate_sample_chunk_size(chunk_size):
    """Normalize an optional positive integer shot chunk size."""
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return chunk_size


def _resolve_measurement_basis(basis, L, *, rng=None):
    """Normalize a global, per-site, or random Pauli basis specification."""
    if basis is None:
        basis = "Z"

    if isinstance(basis, str):
        text = basis.strip().upper().replace(" ", "")
        if text == "RANDOM":
            if rng is None:
                rng = np.random.default_rng()
            return tuple(rng.choice(("X", "Y", "Z"), size=L).tolist())
        if "," in text:
            values = tuple(part for part in text.split(",") if part)
        elif len(text) == 1:
            values = (text,) * L
        else:
            values = tuple(text)
    else:
        try:
            values = tuple(str(value).strip().upper() for value in basis)
        except TypeError as exc:
            raise TypeError(
                "basis must be 'X', 'Y', 'Z', 'random', or a length-L sequence"
            ) from exc

    if len(values) != L or any(value not in _MEASUREMENT_BASIS_LABELS for value in values):
        raise ValueError(
            "basis must be one of 'X', 'Y', 'Z', 'random', or a per-site "
            f"sequence of exactly L={L} entries from X/Y/Z; got {basis!r}."
        )
    return values


def _measurement_rotation(label, *, like=None):
    """Return the measurement-to-Z rotation on ``like``'s backend."""
    if label == "Z":
        rotation = np.eye(2, dtype=complex)
    else:
        hadamard = np.asarray(
            ((1.0, 1.0), (1.0, -1.0)),
            dtype=complex,
        ) / np.sqrt(2.0)
        if label == "X":
            rotation = hadamard
        else:
            # For Y, apply S^dagger followed by H: H S^dagger maps Y
            # eigenstates to computational Z eigenstates.
            phase_dagger = np.diag((1.0, -1.0j))
            rotation = hadamard @ phase_dagger

    if like is None:
        return rotation
    backend = _mps_array_backend(like)
    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        return torch.as_tensor(
            rotation,
            dtype=like.dtype,
            device=like.device,
        )
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        return cp.asarray(rotation, dtype=like.dtype)
    return np.asarray(rotation, dtype=getattr(like, "dtype", complex))


def _basis_selection_probability(basis, L):
    """Return the proposal probability of a resolved basis pattern.

    Explicit basis specifications are treated as deterministic choices. The
    special ``"random"`` policy chooses each site's X/Y/Z label uniformly and
    independently, so one resolved pattern has probability ``3**(-L)``.
    """
    if isinstance(basis, str):
        text = basis.strip().upper().replace(" ", "")
        if text == "RANDOM":
            return float(3.0 ** (-int(L)))
    return 1.0


def _normalize_symmray_prefix_strategy(strategy):
    if strategy is None:
        return "auto"
    key = str(strategy).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "prefix": "prefix",
        "shared_prefix": "prefix",
        "serial": "serial",
        "one_by_one": "serial",
        "dense": "dense",
        "dense_batch": "dense",
        "batched_dense": "dense",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            "Unknown Symmray prefix strategy "
            f"{strategy!r}. Expected one of: {allowed}."
        ) from exc


def _normalize_dense_memory_limit(limit):
    """Normalize a dense sampling memory budget to bytes."""
    if limit is None:
        return None
    if isinstance(limit, (int, np.integer)):
        limit = int(limit)
    else:
        text = str(limit).strip().upper().replace(" ", "")
        if text in {"NONE", "UNBOUNDED", "INF", "INFINITY"}:
            return None
        units = (
            ("GIB", 1024**3),
            ("GB", 1000**3),
            ("MIB", 1024**2),
            ("MB", 1000**2),
            ("KIB", 1024),
            ("KB", 1000),
            ("B", 1),
        )
        multiplier = 1
        for suffix, factor in units:
            if text.endswith(suffix):
                text = text[:-len(suffix)]
                multiplier = factor
                break
        try:
            limit = int(float(text) * multiplier)
        except ValueError as exc:
            raise TypeError(
                "dense_memory_limit must be bytes, a size such as '256MiB', "
                "or None."
            ) from exc
    if limit < 1:
        raise ValueError("dense_memory_limit must be positive or None.")
    return int(limit)


def _mps_array_backend(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "torch":
        return "torch"
    if module == "cupy":
        return "cupy"
    if isinstance(array, np.ndarray):
        return "numpy"
    if hasattr(array, "blocks"):
        return "symmray"
    return "unknown"


def _backend_array_to_numpy(array):
    if hasattr(array, "to_dense"):
        array = array.to_dense()
    return np.asarray(ar.to_numpy(array))


def _fermion_symmray_occupations(charge, offset, fermion):
    """Decode one Symmray physical code into on-site occupations.

    A physical code is an index into a tensor leg, not a universal fermion
    label. In particular, collapsed U1/Z2 sectors retain an offset within a
    charge sector. Keeping this conversion next to the sampler avoids mixing
    the U1U1 and parity-code conventions at VMC boundaries.
    """
    symmetry = str(fermion.symmetry).upper()
    spinful = bool(fermion.spinful)
    if not spinful:
        if symmetry not in {"U1", "Z2"}:
            raise ValueError(
                "Spinless fermion sampling requires symmetry='U1' or 'Z2'."
            )
        occupation = int(charge)
        if occupation not in {0, 1} or int(offset) != 0:
            raise ValueError(
                f"Unexpected spinless {symmetry} physical sector "
                f"{(charge, offset)!r}."
            )
        return (occupation,)

    offset = int(offset)
    if symmetry == "Z2":
        charge = int(charge)
        if charge == 0:
            if offset == 0:
                return (0, 0)
            if offset == 1:
                return (1, 1)
        elif charge == 1:
            # Symmray's parity-collapsed physical basis is empty, double,
            # up, down. This differs from the resolved U1/U1U1 ordering.
            if offset == 0:
                return (1, 0)
            if offset == 1:
                return (0, 1)
        raise ValueError(
            "Spinful Z2 physical sectors must be empty/double or up/down "
            f"pairs; got {(charge, offset)!r}."
        )

    if symmetry == "U1":
        occupation = int(charge)
        if occupation == 0 and offset == 0:
            return (0, 0)
        if occupation == 1:
            if offset == 0:
                return (0, 1)
            if offset == 1:
                return (1, 0)
        if occupation == 2 and offset == 0:
            return (1, 1)
        raise ValueError(
            "Spinful U1 physical sectors must be empty, down/up, or double; "
            f"got {(charge, offset)!r}."
        )

    if symmetry in {"U1U1", "Z2Z2"}:
        occupation = tuple(int(value) for value in charge)
        if len(occupation) != 2 or any(value not in {0, 1} for value in occupation):
            raise ValueError(
                f"Unexpected spinful {symmetry} physical charge {charge!r}."
            )
        if offset != 0:
            raise ValueError(
                f"Spinful {symmetry} physical sectors must not be degenerate."
            )
        return occupation

    raise ValueError(
        "Unsupported Fermion symmetry for sampled physical-code decoding: "
        f"{symmetry!r}."
    )


def _fermion_code_order(encoding, *, default=(0, 1, 2, 3)):
    """Return physical codes ordered by ``(n_up, n_down)`` bits."""
    if encoding is None:
        codes = {
            "empty": int(default[0]),
            "down": int(default[1]),
            "up": int(default[2]),
            "double": int(default[3]),
        }
    else:
        try:
            codes = {
                "empty": int(encoding.empty),
                "double": int(encoding.double),
                "up": int(encoding.up),
                "down": int(encoding.down),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "encoding must expose integer empty, double, up, and down "
                "codes, for example FermionSiteEncoding."
            ) from exc
    values = (
        codes["empty"],
        codes["down"],
        codes["up"],
        codes["double"],
    )
    if sorted(values) != [0, 1, 2, 3]:
        raise ValueError(
            "A spinful fermion encoding must contain exactly the physical "
            "codes 0, 1, 2, and 3."
        )
    return values


def _infer_fermion_code_order(tn, sites):
    """Infer a four-state code order from PEPS symmetry metadata when possible."""
    default = (0, 1, 2, 3)
    if not sites:
        return default
    tensor = tn[sites[0]]
    data = getattr(tensor, "data", None)
    symmetry = str(getattr(data, "symmetry", "")).upper()
    if symmetry == "Z2":
        # The two even states precede the two odd states in Symmray's
        # charge-collapsed physical index.
        return (0, 3, 2, 1)
    if symmetry not in {"U1U1", "Z2Z2"}:
        return default
    try:
        site_ind = tn.site_ind(sites[0])
        axis = tensor.inds.index(site_ind)
        charges = tuple(data.indices[axis].chargemap)
        position = {tuple(charge): i for i, charge in enumerate(charges)}
        required = ((0, 0), (0, 1), (1, 0), (1, 1))
        if all(charge in position for charge in required):
            return tuple(position[charge] for charge in required)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    return default


def _to_dense_numpy(array):
    """Convert a dense or Symmray array to a NumPy array for BP sampling."""
    if hasattr(array, "to_dense"):
        array = array.to_dense()
    return np.asarray(ar.to_numpy(array))


def _prepare_bp_binary_network(tn, *, site_order=None, encoding=None):
    """Prepare a binary-output copy for Quimb's binary ``sample_d2bp``.

    Quimb's public D2BP sampler currently samples each output index from
    ``[0, 1]``. A four-state fermionic physical leg is therefore represented
    as two occupation legs, while all tensor data are converted to dense
    NumPy arrays in the private BP copy. The source network is never mutated.
    """
    network = getattr(tn, "tn", tn)
    if not hasattr(network, "sites") or not hasattr(network, "copy"):
        return network, {}, tuple(), (0, 1, 2, 3)

    sites = tuple(network.sites if site_order is None else site_order)
    missing = [site for site in sites if site not in network.sites]
    if missing:
        raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")

    work = network.copy()
    code_order = (
        _fermion_code_order(encoding)
        if encoding is not None
        else _infer_fermion_code_order(network, sites)
    )
    split_inds = {}
    existing_inds = set(work.ind_map)

    for site in sites:
        tensor = work[site]
        site_ind = work.site_ind(site)
        try:
            axis = tensor.inds.index(site_ind)
        except ValueError as exc:
            raise ValueError(
                f"Could not locate physical index for PEPS site {site!r}."
            ) from exc

        data = _to_dense_numpy(tensor.data)
        physical_dim = int(data.shape[axis])
        if physical_dim == 2:
            if hasattr(tensor.data, "to_dense"):
                tensor.modify(data=data)
            continue
        if physical_dim != 4:
            raise ValueError(
                "Quimb BP sampling supports binary physical legs directly "
                "and spinful fermion legs through a four-state adapter; got "
                f"dimension {physical_dim} at site {site!r}."
            )

        # ``code_order`` maps flattened (up, down) bits back to the PEPS
        # physical-index order. Reshaping after this permutation gives BP two
        # binary output indices with the intended fermion convention.
        data = np.take(data, code_order, axis=axis)
        data = data.reshape(
            data.shape[:axis] + (2, 2) + data.shape[axis + 1:]
        )
        up_ind = f"{site_ind}_bp_up"
        down_ind = f"{site_ind}_bp_down"
        suffix = 0
        while up_ind in existing_inds or down_ind in existing_inds:
            suffix += 1
            up_ind = f"{site_ind}_bp_up_{suffix}"
            down_ind = f"{site_ind}_bp_down_{suffix}"
        existing_inds.update((up_ind, down_ind))
        inds = list(tensor.inds)
        inds[axis:axis + 1] = [up_ind, down_ind]
        tensor.modify(data=data, inds=inds)
        split_inds[site] = (up_ind, down_ind)

    return work, split_inds, sites, code_order


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


class MpsSampler:
    """Sample from an MPS using quimb or a backend-native batched sampler.

    The legacy ``backend="quimb"`` path handles GPU→CPU conversion and calls
    quimb's canonical-form sampler. ``backend="native"`` keeps dense NumPy,
    Torch, or CuPy MPS arrays on their current device, builds right
    environments once, and draws all requested samples with batched conditional
    contractions.

    Parameters
    ----------
    psi : MatrixProductState
        The MPS to sample from (can be on any backend).
    one_d_to_two_d : dict[int, tuple[int, int]], optional
        Mapping from 1D site index to (x, y) lattice coordinate. When omitted,
        a trivial single-row 1D layout ``{i: (i, 0)}`` inferred from the MPS
        length is used, so a plain 1D chain can be sampled without a 2D map.
        backend : {"quimb", "native", "auto", "numpy", "torch", "cupy", "symmray"}
        Sampling implementation. ``"quimb"`` preserves the historical CPU
        behavior for dense MPSs. Symmray-backed MPSs are detected and use the
        native block-sparse sampler rather than being densified. ``"native"``
        accepts dense NumPy/Torch/CuPy tensors and Symmray tensors, while
        ``"symmray"`` requires a Symmray MPS explicitly. ``"auto"`` tries a
        native sampler and falls back to ``"quimb"`` when the MPS layout is
        unsupported.
    torch_compile : bool, default=False
        Opt into ``torch.compile`` for repeated, device-resident, unseeded
        Torch inference batches. Unsupported compiler environments and calls
        that need eager-only behavior fall back to eager sampling.
    strategy : {"auto", "prefix", "serial", "dense"}, optional
        Preferred name for the Symmray sampling strategy. ``None`` leaves
        ``prefix_strategy`` in control for backward compatibility.
    prefix_strategy : {"auto", "prefix", "serial", "dense"}, default="auto"
        Symmray batch-sampling strategy. ``"prefix"`` shares a normalized
        block-sparse boundary between equal sampled prefixes; ``"serial"``
        uses one independent left-to-right sweep per shot. ``"auto"`` uses
        prefix sharing until ``max_prefix_groups`` is reached, then
        finishes the remaining branches serially with bounded memory.
        ``"dense"`` creates a temporary dense view of the source MPS and
        uses the backend-native fully batched sampler. ``"auto"`` selects
        dense batching when the sample count and memory budget permit it.
        Dense batching can use more memory than the sparse routes.
    max_prefix_groups : int or None, default=256
        Maximum active Symmray prefix groups before the ``"auto"`` strategy
        switches the remaining suffixes to serial sampling. ``None`` permits
        all distinct prefixes. This has no effect on dense MPS backends.
    dense_memory_limit : int, str, or None, default="256MiB"
        Maximum estimated dense MPS storage allowed by ``strategy="auto"`` or
        ``strategy="dense"``. Strings such as ``"256MiB"`` and ``"1GB"`` are
        accepted. ``None`` disables the guard.
    dense_min_samples : int, default=1024
        Minimum batch size for ``strategy="auto"`` to select dense batching.
    fermion : pepsy.tensors.Fermion, optional
        Fermionic physical-space convention associated with this sampler. When
        supplied, :meth:`sample_batch` attaches its symmetry-aware
        configuration encoding by default, and the fermionic diagonal helpers
        can omit the repeated ``fermion`` argument. A per-call ``fermion=``
        argument remains supported and takes precedence.

    Notes
    -----
    Dense native right environments and Symmray right-canonical copies are
    cached. The Symmray route retains the source physical-code map before
    canonicalization, then samples by slicing one charge-aware local state and
    absorbing it into a block-sparse boundary. Its batched route shares each
    distinct sampled prefix, including when a physical charge sector has
    degeneracy greater than one (for example spinful fermionic Z2 or U1).
    Call :meth:`refresh` after changing the source MPS; otherwise the sampler
    continues to represent its previous tensor data.
    """

    def __init__(
        self,
        psi,
        one_d_to_two_d: dict[int, tuple[int, int]] | None = None,
        *,
        backend: str | None = "quimb",
        torch_compile: bool = False,
        strategy: str | None = None,
        prefix_strategy: str = "auto",
        max_prefix_groups: int | None = 256,
        dense_memory_limit: int | str | None = 256 * 1024**2,
        dense_min_samples: int = 1024,
        fermion=None,
    ):
        if one_d_to_two_d is None:
            inferred_L = getattr(psi, "L", None)
            if inferred_L is None:
                raise ValueError(
                    "one_d_to_two_d is required when the MPS does not expose an "
                    "'L' attribute to infer the 1D chain length."
                )
            # Default to a trivial single-row 1D chain layout.
            one_d_to_two_d = {site: (site, 0) for site in range(int(inferred_L))}
        self._L = _validate_one_d_to_two_d(
            one_d_to_two_d,
            expected_L=getattr(psi, "L", None),
        )
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1
        self.backend = _normalize_mps_sampler_backend(backend)
        if not isinstance(torch_compile, (bool, np.bool_)):
            raise TypeError("torch_compile must be a boolean.")
        self.torch_compile = bool(torch_compile)
        if strategy is not None:
            if prefix_strategy not in (None, "auto"):
                raise ValueError(
                    "Pass either strategy= or prefix_strategy=, not both."
                )
            prefix_strategy = strategy
        self.prefix_strategy = _normalize_symmray_prefix_strategy(prefix_strategy)
        if max_prefix_groups is not None:
            if not isinstance(max_prefix_groups, (int, np.integer)):
                raise TypeError("max_prefix_groups must be a positive integer or None.")
            if int(max_prefix_groups) < 1:
                raise ValueError(
                    "max_prefix_groups must be a positive integer or None."
                )
            max_prefix_groups = int(max_prefix_groups)
        self.max_prefix_groups = max_prefix_groups
        self.dense_memory_limit = _normalize_dense_memory_limit(dense_memory_limit)
        if not isinstance(dense_min_samples, (int, np.integer)):
            raise TypeError("dense_min_samples must be a positive integer.")
        if int(dense_min_samples) < 1:
            raise ValueError("dense_min_samples must be a positive integer.")
        self.dense_min_samples = int(dense_min_samples)
        # ``strategy`` is the preferred public spelling; retain the old
        # attribute for callers that inspect prefix_strategy directly.
        self.strategy = self.prefix_strategy
        self.fermion = fermion
        self.resolved_backend = None
        self._source_psi = None
        self._native_arrays = None
        self._native_site_ops = None
        self._native_inference_site_ops = None
        self._native_amplitude_site_ops = None
        self._evaluation_backend = None
        self._evaluation_arrays = None
        self._evaluation_site_ops = None
        self._symmray_state = None
        self._last_symmray_sampling_stats = None
        self._psi = None
        self._torch_compiled_sample_fns = {}
        self._torch_compile_disabled = False

        self.refresh(psi)

    def _resolve_fermion(self, fermion):
        """Use a call-specific Fermion or the sampler's bound convention."""
        fermion = self.fermion if fermion is None else fermion
        if fermion is None:
            raise TypeError(
                "fermion is required. Pass fermion=... when constructing "
                "MpsSampler or to this call."
            )
        if not all(hasattr(fermion, name) for name in ("spinful", "symmetry")):
            raise TypeError(
                "fermion must be a pepsy.tensors.Fermion instance or expose "
                "spinful and symmetry attributes."
            )
        return fermion

    def refresh(self, psi=None):
        """Refresh cached state from ``psi`` or the original source MPS.

        The native sampler caches tensor views and right environments for
        repeated sampling. Call this method after an MPS is changed in place
        or its tensors are replaced with ``Tensor.modify(...)``. Supplying
        ``psi`` also changes the source MPS, provided its length matches the
        sampler's fixed site map.

        Returns
        -------
        MpsSampler
            This sampler, with all derived state rebuilt lazily on its next
            sampling or evaluation call.
        """
        if psi is None:
            psi = self._source_psi
        if psi is None:
            raise ValueError("refresh requires an MPS before sampler initialization.")

        source_L = getattr(psi, "L", None)
        if source_L is not None and int(source_L) != self._L:
            raise ValueError(
                "Cannot refresh MpsSampler with an MPS of length "
                f"{int(source_L)}; its site map has length {self._L}."
            )
        self._source_psi = psi

        self._native_arrays = None
        self._native_site_ops = None
        self._native_inference_site_ops = None
        self._native_amplitude_site_ops = None
        self._evaluation_backend = None
        self._evaluation_arrays = None
        self._evaluation_site_ops = None
        self._symmray_state = None
        self._last_symmray_sampling_stats = None
        self._psi = None
        self._torch_compiled_sample_fns.clear()
        self._torch_compile_disabled = False

        source_backends = {
            _mps_array_backend(psi[site].data)
            for site in range(int(psi.L))
        }
        if "symmray" in source_backends:
            if source_backends != {"symmray"}:
                raise ValueError(
                    "MPS tensors use mixed dense and Symmray array backends."
                )
            if self.backend in {"quimb", "native", "auto", "symmray"}:
                self._symmray_state = self._prepare_symmray_state(psi)
                self.resolved_backend = "symmray"
                return self
            raise ValueError(
                f"MpsSampler backend={self.backend!r} requested for a Symmray "
                "MPS. Use backend='symmray', 'native', or 'auto'."
            )
        if self.backend == "symmray":
            raise ValueError(
                "MpsSampler backend='symmray' requires Symmray tensor data."
            )

        if self.backend != "quimb":
            try:
                native_backend, native_arrays = self._prepare_native_arrays(psi)
                if (
                    self.backend in {"numpy", "torch", "cupy"}
                    and native_backend != self.backend
                ):
                    raise ValueError(
                        f"MpsSampler backend={self.backend!r} requested, but "
                        f"the MPS tensors use backend {native_backend!r}."
                    )
                self.resolved_backend = native_backend
                self._native_arrays = native_arrays
                return self
            except Exception as exc:
                if self.backend != "auto":
                    raise
                warnings.warn(
                    "MpsSampler backend='auto' could not prepare the native "
                    "sampler and is falling back to Quimb: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.resolved_backend = "quimb"
        # Convert to numpy for quimb sampling compatibility
        self._psi = psi.copy()
        self._psi.apply_to_arrays(
            lambda x: ar.to_numpy(x)
        )
        return self

    def entanglement_entropy(self, cut=None, *, method="svd"):
        """Measure entropy across one bond of the captured source MPS.

        The source state is not mutated. The calculation is delegated to the
        backend-native tensor observable, so selecting ``backend="quimb"`` for
        sampling does not force this diagnostic through the sampler's legacy
        GPU-to-CPU compatibility copy.
        """
        from ..tensors.observables import (  # pylint: disable=import-outside-toplevel
            mps_entanglement_entropy,
        )

        return mps_entanglement_entropy(
            self._source_psi,
            cut=cut,
            method=method,
        )

    @property
    def physical_code_maps(self):
        """Per-site Symmray ``physical_code -> (charge, sector_offset)`` maps.

        The maps describe the source MPS physical basis, including charge
        sectors pruned from the private canonical sampling copy. They are
        ``None`` for a dense MPS, whose physical codes are already ordinary
        positional indices. A fresh set of dictionaries is returned on each
        access so callers cannot mutate sampler state.
        """
        if self._symmray_state is None:
            return None
        return tuple(
            dict(code_map)
            for code_map in self._symmray_state["physical_code_maps"]
        )

    @property
    def symmray_sampling_stats(self):
        """Diagnostics from the most recent Symmray sampling call, if any.

        ``conditional_evaluations`` counts distinct local distributions built,
        which is the useful work reduced by prefix sharing. ``None`` means the
        sampler has not yet taken the Symmray route.
        """
        if self._last_symmray_sampling_stats is None:
            return None
        return dict(self._last_symmray_sampling_stats)

    def _get_evaluation_arrays(self):
        if self._native_arrays is not None:
            return self.resolved_backend, self._native_arrays
        if self._evaluation_arrays is None:
            self._evaluation_backend, self._evaluation_arrays = (
                self._prepare_native_arrays(self._psi)
            )
        return self._evaluation_backend, self._evaluation_arrays

    def _get_native_site_ops(self, *, track_grad=True):
        if self.resolved_backend == "torch" and not track_grad:
            if self._native_inference_site_ops is None:
                import torch  # pylint: disable=import-outside-toplevel

                arrays = tuple(array.detach() for array in self._native_arrays)
                with torch.no_grad():
                    self._native_inference_site_ops = self._prepare_site_ops(
                        self.resolved_backend,
                        arrays,
                    )
            return self.resolved_backend, self._native_inference_site_ops
        if self._native_site_ops is None:
            self._native_site_ops = self._prepare_site_ops(
                self.resolved_backend,
                self._native_arrays,
            )
        return self.resolved_backend, self._native_site_ops

    def _get_evaluation_site_ops(self):
        if self._native_arrays is not None:
            return self._get_native_site_ops(track_grad=True)
        backend, arrays = self._get_evaluation_arrays()
        if self._evaluation_site_ops is None:
            self._evaluation_site_ops = self._prepare_site_ops(backend, arrays)
        return backend, self._evaluation_site_ops

    @staticmethod
    def _site_array_lr_phys_r(psi, site):
        tensor = psi[site]
        site_ind = psi.site_ind(site)
        left_ind = psi.bond(site - 1, site) if site > 0 else None
        right_ind = psi.bond(site, site + 1) if site < psi.L - 1 else None
        if left_ind is None and right_ind is None:
            data = tensor.transpose(site_ind).data
            return data.reshape((1, data.shape[0], 1))
        if left_ind is None:
            data = tensor.transpose(site_ind, right_ind).data
            return data.reshape((1, data.shape[0], data.shape[1]))
        if right_ind is None:
            data = tensor.transpose(left_ind, site_ind).data
            return data.reshape((data.shape[0], data.shape[1], 1))
        return tensor.transpose(left_ind, site_ind, right_ind).data

    def _prepare_native_arrays(self, psi):
        arrays = tuple(
            self._site_array_lr_phys_r(psi, site)
            for site in range(psi.L)
        )
        if not arrays:
            raise ValueError("Cannot sample an empty MPS.")
        backends = {_mps_array_backend(array) for array in arrays}
        if len(backends) != 1:
            raise ValueError(f"MPS tensors use mixed backends {sorted(backends)!r}.")
        backend = next(iter(backends))
        if backend not in {"numpy", "torch", "cupy"}:
            raise ValueError(
                "backend-native MPS sampling currently supports dense NumPy, "
                f"Torch, or CuPy arrays, not {backend!r}."
            )
        return backend, arrays

    @staticmethod
    def _prepare_symmray_state(psi):
        """Cache a canonical Symmray MPS without altering its physical basis."""
        if getattr(psi, "cyclic", False):
            raise ValueError("Symmray MPS sampling currently requires an open chain.")

        try:
            import symmray as sr  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - guarded by data type
            raise ImportError(
                "Symmray tensor data requires the optional 'symmray' package."
            ) from exc

        source_data = tuple(psi[site].data for site in range(psi.L))
        block_backends = {str(data.backend) for data in source_data}
        if len(block_backends) != 1:
            raise ValueError(
                "Symmray MPS tensors must use one common underlying block backend; "
                f"got {sorted(block_backends)!r}."
            )
        array_backend = next(iter(block_backends))
        if array_backend not in {"numpy", "torch", "cupy"}:
            raise ValueError(
                "Symmray MPS sampling currently supports NumPy, Torch, or CuPy "
                f"blocks, not {array_backend!r}."
            )

        # Quimb's canonicalization runs Symmray QR/SVD blockwise. It can prune
        # identically-zero physical charge sectors, so retain the source basis
        # map below and never expose the canonical copy as the input state.
        canonical = psi.right_canonicalize(normalize=True)
        sites = []
        physical_code_maps = []
        for site in range(psi.L):
            source_tensor = psi[site]
            tensor = canonical[site]
            source_phys_ind = psi.site_ind(site)
            phys_ind = canonical.site_ind(site)
            source_axis = source_tensor.inds.index(source_phys_ind)
            phys_axis = tensor.inds.index(phys_ind)
            source_index = source_tensor.data.indices[source_axis]
            phys_index = tensor.data.indices[phys_axis]

            source_offsets = {}
            source_code_metadata = []
            offset = 0
            for charge, size in source_index.chargemap.items():
                source_offsets[charge] = offset
                for sector_offset in range(int(size)):
                    source_code_metadata.append((charge, sector_offset))
                offset += int(size)

            code_map = []
            for charge, size in phys_index.chargemap.items():
                try:
                    source_size = int(source_index.chargemap[charge])
                except KeyError as exc:
                    raise ValueError(
                        "Canonical Symmray MPS changed a physical charge sector "
                        f"at site {site}: {charge!r}."
                    ) from exc
                size = int(size)
                if size > source_size:
                    raise ValueError(
                        "Canonical Symmray MPS enlarged a physical charge sector "
                        f"at site {site}: {charge!r}."
                    )
                code_map.extend(range(source_offsets[charge], source_offsets[charge] + size))
            if len(code_map) != int(tensor.data.shape[phys_axis]):
                raise ValueError(
                    "Could not reconstruct the physical code map for canonical "
                    f"Symmray site {site}."
                )

            left_axis = None
            if site:
                remaining_inds = tuple(ind for ind in tensor.inds if ind != phys_ind)
                left_axis = remaining_inds.index(canonical.bond(site - 1, site))

            # Physical selection on a fermionic Symmray array must happen
            # before absorbing the left boundary: selecting it afterwards can
            # lose the dummy-mode ordering needed for an odd physical leg.
            # Cache these immutable local branch tensors once, so every
            # sampled prefix shares both the slices and their block metadata.
            locals_ = []
            local_left_charges = []
            nonempty_local_codes = []
            for local_code in range(int(tensor.data.shape[phys_axis])):
                item = [slice(None)] * tensor.data.ndim
                item[phys_axis] = local_code
                local = tensor.data[tuple(item)]
                locals_.append(local)
                blocks = getattr(local, "blocks", None)
                if isinstance(blocks, dict):
                    nonempty = bool(blocks)
                else:
                    nonempty = True
                if nonempty:
                    nonempty_local_codes.append(local_code)

                charges = None
                if left_axis is not None:
                    if isinstance(blocks, dict):
                        charges = frozenset(
                            sector[left_axis] for sector in blocks
                        )
                    else:
                        try:
                            charges = frozenset(
                                local.indices[left_axis].chargemap
                            )
                        except (AttributeError, IndexError, TypeError):
                            charges = None
                local_left_charges.append(charges)
            sites.append(
                {
                    "data": tensor.data,
                    "phys_axis": phys_axis,
                    "left_axis": left_axis,
                    "codes": tuple(code_map),
                    "code_metadata": tuple(
                        source_code_metadata[code] for code in code_map
                    ),
                    "code_to_local": {code: local for local, code in enumerate(code_map)},
                    "locals": tuple(locals_),
                    "nonempty_local_codes": tuple(nonempty_local_codes),
                    "local_left_charges": tuple(local_left_charges),
                }
            )
            physical_code_maps.append(dict(enumerate(source_code_metadata)))

        template = source_data[0].get_any_array()
        # Keep the import local so Symmray remains optional for ordinary MPSs.
        return {
            "sr": sr,
            "mps": canonical,
            "source_mps": psi,
            "sites": tuple(sites),
            "physical_code_maps": tuple(physical_code_maps),
            "array_backend": array_backend,
            "template": template,
            "dense_site_data": None,
            "dense_code_maps": None,
        }

    @staticmethod
    def _symmray_weight(value, state):
        """Return ``||value||**2`` without converting an array to dense.

        Symmray returns an ordinary scalar/block when a contraction has no
        remaining charge structure. This is common for the final local branch
        of fermionic Z2 and U1 MPSs with degenerate physical sectors. Such a
        block is already a selected sector rather than a densified state.
        """
        blocks = getattr(value, "blocks", None)
        if isinstance(blocks, dict) and not blocks:
            return 0.0
        if np.isscalar(value):
            return abs(value) ** 2
        backend = _mps_array_backend(value)
        if backend == "torch":
            return (value.conj() * value).sum().real
        if backend == "cupy":
            return (value.conj() * value).sum().real
        if backend == "numpy":
            return np.sum(np.abs(value) ** 2).real
        return state["sr"].linalg.norm(value) ** 2

    @staticmethod
    def _symmray_scalar(value):
        """Extract a scalar after a fully contracted Symmray MPS branch."""
        blocks = getattr(value, "blocks", None)
        if isinstance(blocks, dict) and not blocks:
            return 0.0
        if hasattr(value, "get_scalar_element"):
            return value.phase_sync().get_scalar_element()
        return value

    @staticmethod
    def _symmray_distribution(weights, state):
        """Normalize scalar weights on the backend of the Symmray blocks."""
        backend = state["array_backend"]
        template = state["template"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            dtype = template.real.dtype
            values = torch.stack([
                torch.as_tensor(weight, dtype=dtype, device=template.device).real
                for weight in weights
            ])
            values = values.clamp_min(0.0)
            total = values.sum()
            if not bool(torch.isfinite(total).detach().cpu().item()) or not bool(
                (total > 0).detach().cpu().item()
            ):
                raise ValueError("MPS has a zero or non-finite conditional norm.")
            return values / total

        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            values = cp.stack([
                cp.asarray(weight, dtype=template.real.dtype).real
                for weight in weights
            ])
            values = cp.maximum(values, 0.0)
            total = values.sum()
            if not bool(cp.isfinite(total).item()) or not bool((total > 0).item()):
                raise ValueError("MPS has a zero or non-finite conditional norm.")
            return values / total

        values = np.asarray(weights, dtype=np.asarray(template).real.dtype).real
        values = np.maximum(values, 0.0)
        total = values.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("MPS has a zero or non-finite conditional norm.")
        return values / total

    @staticmethod
    def _symmray_draw(probs, state, rng):
        """Draw one local code, leaving probabilities on their native backend."""
        backend = state["array_backend"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            choice = int(torch.multinomial(probs, 1, generator=rng).reshape(()).item())
        elif backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            cdf = cp.cumsum(probs)
            choice = int(cp.sum(rng.random() > cdf).item())
            choice = min(choice, int(probs.shape[0]) - 1)
        else:
            choice = int(rng.choice(len(probs), p=probs))
        return choice, probs[choice]

    @staticmethod
    def _symmray_draw_many(probs, n_draws, state, rng):
        """Draw a prefix group and return its local choices on the host.

        The probability vector and random-number generation stay on the
        Symmray block backend. Only the integer decisions cross to Python so
        that one block-sparse boundary can be retained for every distinct
        prefix, rather than once per requested shot.
        """
        n_draws = int(n_draws)
        if n_draws < 1:  # pragma: no cover - internal guard
            raise ValueError("n_draws must be positive.")
        if n_draws == 1:
            choice, _ = MpsSampler._symmray_draw(probs, state, rng)
            return np.asarray((choice,), dtype=np.int64)

        backend = state["array_backend"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            choices = torch.multinomial(
                probs,
                n_draws,
                replacement=True,
                generator=rng,
            )
            return np.asarray(ar.to_numpy(choices), dtype=np.int64)
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            cdf = cp.cumsum(probs)
            draws = rng.random(n_draws)
            choices = cp.searchsorted(cdf, draws, side="right")
            choices = cp.minimum(choices, int(probs.shape[0]) - 1)
            return np.asarray(ar.to_numpy(choices), dtype=np.int64)
        return np.asarray(
            rng.choice(len(probs), size=n_draws, p=probs),
            dtype=np.int64,
        )

    @staticmethod
    def _symmray_rng(state, seed):
        """Create the random generator associated with the block backend."""
        backend = state["array_backend"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            if seed is None:
                return None
            generator = torch.Generator(device=state["template"].device)
            generator.manual_seed(int(seed))
            return generator
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            return cp.random.default_rng(seed)
        return np.random.default_rng(seed)

    @staticmethod
    def _symmray_sqrt(value, state):
        backend = state["array_backend"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            return torch.sqrt(value)
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            return cp.sqrt(value)
        return np.sqrt(value)

    @staticmethod
    def _symmray_positive(value, state):
        """Test a scalar branch norm without moving tensor data to dense CPU."""
        backend = state["array_backend"]
        if backend == "torch":
            positive = value > 0
            return bool(
                positive.detach().cpu().item()
                if hasattr(positive, "detach")
                else positive
            )
        if backend == "cupy":
            positive = value > 0
            return bool(positive.item() if hasattr(positive, "item") else positive)
        return bool(value > 0)

    @staticmethod
    def _symmray_one(state, *, complex_value=False):
        backend = state["array_backend"]
        template = state["template"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            dtype = template.dtype if complex_value else template.real.dtype
            return torch.ones((), dtype=dtype, device=template.device)
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            dtype = template.dtype if complex_value else template.real.dtype
            return cp.ones((), dtype=dtype)
        dtype = np.asarray(template).dtype
        if not complex_value:
            dtype = np.asarray(template).real.dtype
        return np.ones((), dtype=dtype)

    @classmethod
    def _symmray_sample_one(cls, state, rng):
        """Draw one configuration by slice-and-absorb on a canonical MPS."""
        boundary = None
        config = []
        probability = cls._symmray_one(state)
        for site, site_state in enumerate(state["sites"]):
            site_state, local_codes, candidates, weights = cls._symmray_candidates(
                state,
                site,
                boundary,
            )
            probs = cls._symmray_distribution(weights, state)
            choice_index, choice_prob = cls._symmray_draw(probs, state, rng)
            local_code = local_codes[choice_index]
            config.append(site_state["codes"][local_code])
            probability = probability * choice_prob
            if site < len(state["sites"]) - 1:
                boundary = candidates[choice_index] / cls._symmray_sqrt(
                    weights[choice_index],
                    state,
                )
        return config, probability

    @staticmethod
    def _symmray_boundary_charges(boundary):
        """Return the possible outgoing charge labels of a prefix boundary."""
        if boundary is None:
            return None
        blocks = getattr(boundary, "blocks", None)
        if isinstance(blocks, dict):
            return frozenset(sector[0] for sector in blocks)
        try:
            return frozenset(boundary.indices[0].chargemap)
        except (AttributeError, IndexError, TypeError):
            return None

    @classmethod
    def _symmray_candidate_codes(cls, site_state, boundary):
        """Skip cached local branches incompatible with the prefix charge."""
        local_codes = site_state["nonempty_local_codes"]
        if boundary is None or site_state["left_axis"] is None:
            return local_codes
        boundary_charges = cls._symmray_boundary_charges(boundary)
        if not boundary_charges:
            return local_codes
        return tuple(
            local_code
            for local_code in local_codes
            if (
                site_state["local_left_charges"][local_code] is None
                or boundary_charges
                & site_state["local_left_charges"][local_code]
            )
        )

    @classmethod
    def _symmray_candidates(cls, state, site, boundary):
        """Build one charge-pruned conditional from cached local branches."""
        site_state = state["sites"][site]
        local_codes = cls._symmray_candidate_codes(site_state, boundary)
        candidates = []
        weights = []
        for local_code in local_codes:
            local = site_state["locals"][local_code]
            if boundary is None:
                candidate = local
            else:
                candidate = state["sr"].tensordot(
                    boundary,
                    local,
                    axes=((0,), (site_state["left_axis"],)),
                )
            candidates.append(candidate)
            weights.append(cls._symmray_weight(candidate, state))
        return site_state, local_codes, candidates, weights

    @staticmethod
    def _dense_array_nbytes(array):
        """Estimate dense storage for a backend array without copying it."""
        nbytes = getattr(array, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        try:
            return int(array.numel()) * int(array.element_size())
        except (AttributeError, TypeError, ValueError):
            return int(np.asarray(array).nbytes)

    @classmethod
    def _symmray_estimate_dense_site_bytes(cls, state):
        """Estimate dense MPS storage from shapes without materializing it."""
        template = state["template"]
        if hasattr(template, "element_size"):
            itemsize = int(template.element_size())
        else:
            itemsize = int(np.dtype(getattr(template, "dtype", template)).itemsize)
        total = 0
        source_mps = state["source_mps"]
        for site in range(len(state["sites"])):
            array = cls._site_array_lr_phys_r(source_mps, site)
            total += int(np.prod(array.shape)) * itemsize
        return int(total)

    def _resolve_symmray_sampling_strategy(self, n_samples):
        """Resolve the requested strategy before any dense allocation."""
        requested = self.prefix_strategy
        state = self._require_symmray_state()
        symmetry = str(state["sites"][0]["data"].symmetry).upper()
        dense_supported = symmetry in {"U1", "U1U1"}
        if not dense_supported:
            if requested == "dense":
                raise ValueError(
                    "Dense Symmray sampling is supported only for resolved "
                    f"U1/U1U1 states, not symmetry={symmetry!r}. Use "
                    "strategy='prefix' for charge-aware sampling."
                )
            if requested == "auto":
                return "auto", "auto_sparse_unsupported_symmetry", None
            return requested, "explicit_sparse", None
        estimated_bytes = self._symmray_estimate_dense_site_bytes(state)
        if requested == "dense":
            if (
                self.dense_memory_limit is not None
                and estimated_bytes > self.dense_memory_limit
            ):
                raise ValueError(
                    "Dense Symmray sampling requires an estimated "
                    f"{estimated_bytes} bytes, above the configured limit of "
                    f"{self.dense_memory_limit} bytes. Increase "
                    "dense_memory_limit or use strategy='prefix'."
                )
            return "dense", "explicit_dense", estimated_bytes
        if requested == "auto":
            if (
                int(n_samples) >= self.dense_min_samples
                and (
                    self.dense_memory_limit is None
                    or estimated_bytes <= self.dense_memory_limit
                )
            ):
                return "dense", "auto_dense_within_budget", estimated_bytes
            return "auto", "auto_sparse_fallback", estimated_bytes
        return requested, "explicit_sparse", estimated_bytes

    @classmethod
    def _symmray_dense_site_data(cls, state):
        """Prepare cached dense site operators for explicit dense batching.

        This route is deliberately opt-in. It keeps the source and canonical
        Symmray states intact, materializing only a private sampling view so
        the dense native sampler can contract every shot in one backend batch.
        """
        cached = state.get("dense_site_data")
        if cached is not None:
            return cached

        arrays = []
        source_mps = state["source_mps"]
        for site in range(len(state["sites"])):
            # Use the source MPS rather than the private canonical copy here.
            # Symmray's fermionic bond orientations can have different dense
            # positional layouts on dual virtual legs even though sparse
            # charge-aware contractions remain valid. The source chain has
            # matching virtual dimensions, so its dense view is unambiguous.
            array = cls._site_array_lr_phys_r(source_mps, site)
            if hasattr(array, "to_dense"):
                array = array.to_dense()
            arrays.append(array)

        backends = {_mps_array_backend(array) for array in arrays}
        if len(backends) != 1:
            raise ValueError(
                "Dense Symmray sampling requires one common dense backend; "
                f"got {sorted(backends)!r}."
            )
        backend = next(iter(backends))
        if backend == "torch":
            site_data = cls._torch_site_ops(tuple(arrays))
        elif backend in {"numpy", "cupy"}:
            site_data = cls._array_namespace_site_ops(
                tuple(arrays),
                backend=backend,
            )
        else:
            raise ValueError(
                "Dense Symmray sampling produced unsupported arrays "
                f"with backend {backend!r}."
            )

        dense_bytes = sum(cls._dense_array_nbytes(array) for array in arrays)
        code_maps = tuple(
            tuple(range(int(array.shape[1])))
            for array in arrays
        )
        cached = (backend, site_data, int(dense_bytes))
        state["dense_site_data"] = cached
        state["dense_code_maps"] = code_maps
        return cached

    @staticmethod
    def _symmray_map_dense_configs(configs, state):
        """Map canonical dense physical choices back to source code labels."""
        backend = state["array_backend"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            mapped = torch.empty_like(configs)
            for site, code_map in enumerate(state["dense_code_maps"]):
                lookup = torch.as_tensor(
                    code_map,
                    dtype=torch.long,
                    device=configs.device,
                )
                mapped[:, site] = lookup[configs[:, site]]
            return mapped
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            mapped = cp.empty_like(configs)
            for site, code_map in enumerate(state["dense_code_maps"]):
                lookup = cp.asarray(code_map, dtype=cp.int64)
                mapped[:, site] = lookup[configs[:, site]]
            return mapped

        mapped = np.empty_like(configs)
        for site, code_map in enumerate(state["dense_code_maps"]):
            mapped[:, site] = np.asarray(code_map, dtype=np.int64)[
                configs[:, site]
            ]
        return mapped

    @classmethod
    def _symmray_sample_arrays_dense(
        cls,
        state,
        n_samples,
        seed,
        *,
        to_numpy,
    ):
        """Sample a Symmray MPS with the dense native batched kernels."""
        backend, site_data, dense_bytes = cls._symmray_dense_site_data(state)
        if backend == "torch":
            canonical_configs, probabilities = cls._torch_sample(
                site_data,
                int(n_samples),
                seed,
                to_numpy=False,
            )
        else:
            canonical_configs, probabilities = cls._array_namespace_sample(
                site_data,
                int(n_samples),
                seed,
                backend=backend,
                to_numpy=False,
            )
        configs = cls._symmray_map_dense_configs(canonical_configs, state)
        stats = {
            "strategy": "dense",
            "n_samples": int(n_samples),
            "conditional_evaluations": len(state["sites"]),
            "candidate_contractions": sum(
                len(site_state["codes"]) for site_state in state["sites"]
            ),
            "static_pruned_branches": 0,
            "charge_pruned_branches": 0,
            "cached_local_slices": False,
            "max_active_prefix_groups": 1,
            "serial_fallback": False,
            "adaptive_serial_fallback": False,
            "dense_site_bytes": int(dense_bytes),
            "dense_batch_width": int(n_samples),
        }
        if to_numpy:
            configs = _backend_array_to_numpy(configs)
            probabilities = _backend_array_to_numpy(probabilities)
        return configs, probabilities, stats

    @staticmethod
    def _symmray_note_candidates(stats, site_state, local_codes):
        """Record the sparse branch work avoided by cache/pruning."""
        if stats is None:
            return
        stats["candidate_contractions"] += len(local_codes)
        stats["static_pruned_branches"] += (
            len(site_state["codes"]) - len(site_state["nonempty_local_codes"])
        )
        stats["charge_pruned_branches"] += (
            len(site_state["nonempty_local_codes"]) - len(local_codes)
        )

    @staticmethod
    def _symmray_stack(values, state, *, integer=False, complex_value=False):
        """Stack scalar results using the backend of the Symmray blocks."""
        backend = state["array_backend"]
        template = state["template"]
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            if integer:
                return torch.as_tensor(values, dtype=torch.long, device=template.device)
            dtype = template.dtype if complex_value else template.real.dtype
            return torch.stack([
                torch.as_tensor(value, dtype=dtype, device=template.device)
                for value in values
            ])
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            dtype = (
                cp.int64
                if integer
                else (template.dtype if complex_value else template.real.dtype)
            )
            return cp.asarray(values, dtype=dtype)
        dtype = (
            np.int64
            if integer
            else (
                np.asarray(template).dtype
                if complex_value
                else np.asarray(template).real.dtype
            )
        )
        return np.asarray(values, dtype=dtype)

    @classmethod
    def _symmray_sample_from_prefix(
        cls,
        state,
        rng,
        config,
        probability,
        boundary,
        start_site,
        stats,
    ):
        """Complete one shot from an already sampled normalized prefix."""
        for site in range(start_site, len(state["sites"])):
            site_state, local_codes, candidates, weights = cls._symmray_candidates(
                state,
                site,
                boundary,
            )
            stats["conditional_evaluations"] += 1
            cls._symmray_note_candidates(stats, site_state, local_codes)
            probs = cls._symmray_distribution(weights, state)
            choice_index, choice_prob = cls._symmray_draw(probs, state, rng)
            local_code = local_codes[choice_index]
            config.append(site_state["codes"][local_code])
            probability = probability * choice_prob
            if site < len(state["sites"]) - 1:
                if not cls._symmray_positive(weights[choice_index], state):
                    raise ValueError(
                        "MPS sampler selected a zero-norm conditional branch."
                    )
                boundary = candidates[choice_index] / cls._symmray_sqrt(
                    weights[choice_index],
                    state,
                )
        return config, probability

    @classmethod
    def _symmray_sample_arrays_serial(cls, state, n_samples, seed, *, to_numpy):
        """Sample independently, using constant block-sparse boundary memory."""
        rng = cls._symmray_rng(state, seed)
        configs = []
        probabilities = []
        stats = {
            "strategy": "serial",
            "n_samples": int(n_samples),
            "conditional_evaluations": 0,
            "candidate_contractions": 0,
            "static_pruned_branches": 0,
            "charge_pruned_branches": 0,
            "cached_local_slices": True,
            "max_active_prefix_groups": 1,
            "serial_fallback": False,
            "adaptive_serial_fallback": False,
        }
        for _ in range(int(n_samples)):
            config, probability = cls._symmray_sample_from_prefix(
                state,
                rng,
                [],
                cls._symmray_one(state),
                None,
                0,
                stats,
            )
            configs.append(config)
            probabilities.append(probability)
        return cls._symmray_finalize_samples(
            configs,
            probabilities,
            state,
            stats,
            to_numpy=to_numpy,
        )

    @staticmethod
    def _symmray_boundary_storage_cost(boundary):
        """Estimate block-resident boundary storage without densifying it."""
        blocks = getattr(boundary, "blocks", None)
        if not isinstance(blocks, dict):
            return 1
        cost = 0
        for block in blocks.values():
            size = getattr(block, "size", None)
            if callable(size):
                size = size()
            if size is None or not np.isscalar(size):
                size = int(np.prod(getattr(block, "shape", (1,))))
            cost += int(size)
        return max(cost, 1)

    @classmethod
    def _symmray_select_prefix_groups(
        cls,
        branches,
        *,
        max_prefix_groups,
        adaptive,
    ):
        """Keep prefix boundaries only while they amortize their storage.

        A group with one walker cannot share a future conditional, so the
        auto strategy finishes it serially immediately. For the remaining
        groups, ``max_prefix_groups`` is a hard count cap and also defines a
        per-level block-storage budget relative to the median boundary size.
        This avoids treating a large multi-sector boundary as equal to a tiny
        one-sector boundary.
        """
        if not branches:
            return (), (), None

        costs = [cls._symmray_boundary_storage_cost(branch[1]) for branch in branches]
        if max_prefix_groups is None:
            count_limit = None
            storage_budget = None
        else:
            count_limit = int(max_prefix_groups)
            baseline = int(np.median(costs))
            storage_budget = max(count_limit * max(baseline, 1), 1)

        kept_ids = set()
        used_storage = 0
        # Retain the groups with the greatest prospective reuse first. Ties
        # favor smaller block boundaries, then preserve sample order.
        ranked = sorted(
            range(len(branches)),
            key=lambda index: (
                -len(branches[index][0]),
                costs[index],
                int(branches[index][0][0]),
            ),
        )
        for index in ranked:
            positions = branches[index][0]
            if adaptive and len(positions) == 1:
                continue
            if count_limit is not None and len(kept_ids) >= count_limit:
                continue
            if (
                storage_budget is not None
                and kept_ids
                and used_storage + costs[index] > storage_budget
            ):
                continue
            kept_ids.add(index)
            used_storage += costs[index]

        kept = tuple(branch for index, branch in enumerate(branches) if index in kept_ids)
        dropped = tuple(branch for index, branch in enumerate(branches) if index not in kept_ids)
        return kept, dropped, storage_budget

    @classmethod
    def _symmray_sample_arrays_prefix(
        cls,
        state,
        n_samples,
        seed,
        *,
        max_prefix_groups,
        adaptive,
        to_numpy,
    ):
        """Share boundaries between prefixes, with an optional memory bound."""
        rng = cls._symmray_rng(state, seed)
        n_samples = int(n_samples)
        configs = np.empty((n_samples, len(state["sites"])), dtype=np.int64)
        probabilities = [None] * n_samples
        positions = np.arange(n_samples, dtype=np.int64)
        stats = {
            "strategy": "prefix",
            "n_samples": n_samples,
            "conditional_evaluations": 0,
            "candidate_contractions": 0,
            "static_pruned_branches": 0,
            "charge_pruned_branches": 0,
            "cached_local_slices": True,
            "max_active_prefix_groups": 1,
            "serial_fallback": False,
            "adaptive_serial_fallback": False,
            "max_prefix_storage_budget": None,
        }
        # A group stores the shared selected prefix, its normalized boundary,
        # and its probability. Only current-depth groups are retained.
        groups = [(positions, None, cls._symmray_one(state))]

        for site in range(len(state["sites"])):
            stats["max_active_prefix_groups"] = max(
                stats["max_active_prefix_groups"],
                len(groups),
            )

            is_final_site = site == len(state["sites"]) - 1
            branches = []
            for group_positions, boundary, prefix_probability in groups:
                site_state, local_codes, candidates, weights = cls._symmray_candidates(
                    state,
                    site,
                    boundary,
                )
                stats["conditional_evaluations"] += 1
                cls._symmray_note_candidates(stats, site_state, local_codes)
                probs = cls._symmray_distribution(weights, state)
                choices = cls._symmray_draw_many(
                    probs,
                    len(group_positions),
                    state,
                    rng,
                )
                for choice_index, local_code in enumerate(local_codes):
                    selected = group_positions[choices == choice_index]
                    if not len(selected):
                        continue
                    code = site_state["codes"][local_code]
                    configs[selected, site] = code
                    selected_probability = prefix_probability * probs[choice_index]
                    if is_final_site:
                        for sample in selected:
                            probabilities[int(sample)] = selected_probability
                        continue
                    if not cls._symmray_positive(weights[choice_index], state):
                        raise ValueError(
                            "MPS sampler selected a zero-norm conditional branch."
                        )
                    next_boundary = candidates[choice_index] / cls._symmray_sqrt(
                        weights[choice_index],
                        state,
                    )
                    branches.append((selected, next_boundary, selected_probability))

            if is_final_site:
                groups = ()
                continue

            groups, serial_branches, storage_budget = cls._symmray_select_prefix_groups(
                branches,
                max_prefix_groups=max_prefix_groups,
                adaptive=adaptive,
            )
            if storage_budget is not None:
                previous_budget = stats["max_prefix_storage_budget"]
                stats["max_prefix_storage_budget"] = (
                    storage_budget
                    if previous_budget is None
                    else max(previous_budget, storage_budget)
                )
            if serial_branches:
                stats["serial_fallback"] = True
            for selected, next_boundary, selected_probability in serial_branches:
                if adaptive and len(selected) == 1:
                    stats["adaptive_serial_fallback"] = True
                for sample in selected:
                    config, probability = cls._symmray_sample_from_prefix(
                        state,
                        rng,
                        configs[int(sample), : site + 1].tolist(),
                        selected_probability,
                        next_boundary,
                        site + 1,
                        stats,
                    )
                    configs[int(sample)] = config
                    probabilities[int(sample)] = probability

        if any(probability is None for probability in probabilities):  # pragma: no cover
            raise RuntimeError("Symmray prefix sampler did not assign every shot.")
        return cls._symmray_finalize_samples(
            configs,
            probabilities,
            state,
            stats,
            to_numpy=to_numpy,
        )

    @classmethod
    def _symmray_finalize_samples(
        cls,
        configs,
        probabilities,
        state,
        stats,
        *,
        to_numpy,
    ):
        configs = cls._symmray_stack(configs, state, integer=True)
        probabilities = cls._symmray_stack(probabilities, state)
        if to_numpy:
            configs = _backend_array_to_numpy(configs)
            probabilities = _backend_array_to_numpy(probabilities)
        return configs, probabilities, stats

    @classmethod
    def _symmray_sample_arrays(
        cls,
        state,
        n_samples,
        seed,
        *,
        strategy,
        max_prefix_groups,
        to_numpy,
    ):
        if strategy == "dense":
            return cls._symmray_sample_arrays_dense(
                state,
                n_samples,
                seed,
                to_numpy=to_numpy,
            )
        if strategy == "serial":
            return cls._symmray_sample_arrays_serial(
                state,
                n_samples,
                seed,
                to_numpy=to_numpy,
            )
        return cls._symmray_sample_arrays_prefix(
            state,
            n_samples,
            seed,
            max_prefix_groups=max_prefix_groups,
            adaptive=(strategy == "auto"),
            to_numpy=to_numpy,
        )

    @staticmethod
    def _symmray_config_rows(configs, *, L):
        """Validate discrete configurations for Symmray MPS evaluation."""
        configs = np.asarray(_backend_array_to_numpy(configs), dtype=np.int64)
        if configs.ndim != 2 or configs.shape[1] != int(L):
            raise ValueError(
                f"configs must have shape (batch, L={int(L)}); "
                f"got {tuple(configs.shape)}."
            )
        return configs

    @classmethod
    def _symmray_amplitude_one(cls, state, config):
        """Contract one selected configuration without densifying the MPS."""
        boundary = None
        for site, code in enumerate(config):
            site_state = state["sites"][site]
            try:
                local_code = site_state["code_to_local"][int(code)]
            except KeyError:
                return cls._symmray_one(state, complex_value=True) * 0.0
            local = site_state["locals"][local_code]
            if boundary is None:
                boundary = local
            else:
                boundary = state["sr"].tensordot(
                    boundary,
                    local,
                    axes=((0,), (site_state["left_axis"],)),
                )
        return cls._symmray_scalar(boundary)

    @classmethod
    def _symmray_probability_one(cls, state, config):
        """Evaluate one Born probability with canonical slice-and-absorb."""
        boundary = None
        probability = cls._symmray_one(state)
        for site, code in enumerate(config):
            site_state = state["sites"][site]
            try:
                choice = site_state["code_to_local"][int(code)]
            except KeyError:
                return cls._symmray_one(state) * 0.0

            site_state, local_codes, candidates, weights = cls._symmray_candidates(
                state,
                site,
                boundary,
            )
            try:
                choice_index = local_codes.index(choice)
            except ValueError:
                return cls._symmray_one(state) * 0.0
            probs = cls._symmray_distribution(weights, state)
            probability = probability * probs[choice_index]
            if site < len(state["sites"]) - 1:
                if not cls._symmray_positive(weights[choice_index], state):
                    return cls._symmray_one(state) * 0.0
                boundary = candidates[choice_index] / cls._symmray_sqrt(
                    weights[choice_index],
                    state,
                )
        return probability

    def _symmray_amplitudes(self, configs):
        state = self._require_symmray_state()
        rows = self._symmray_config_rows(configs, L=len(state["sites"]))
        values = [self._symmray_amplitude_one(state, row) for row in rows]
        return self._symmray_stack(values, state, complex_value=True)

    def _symmray_probabilities(self, configs):
        state = self._require_symmray_state()
        rows = self._symmray_config_rows(configs, L=len(state["sites"]))
        values = [self._symmray_probability_one(state, row) for row in rows]
        return self._symmray_stack(values, state)

    def _require_symmray_state(self):
        if self._symmray_state is None:  # pragma: no cover - internal guard
            raise RuntimeError("Symmray sampler state has not been initialized.")
        return self._symmray_state

    def fermion_configuration_encoding(self, fermion=None) -> FermionConfigurationEncoding:
        """Return the physical-code/occupation contract for a fermionic MPS.

        This is the explicit bridge from MPS Born samples to a VMC walker or
        local-estimator configuration. It is intentionally derived from the
        source physical-sector maps retained by the Symmray sampler, rather
        than assuming a dense-basis or VMC-specific code convention.
        """
        fermion = self._resolve_fermion(fermion)
        state = self._require_symmray_state()
        state_symmetry = str(state["sites"][0]["data"].symmetry).upper()
        symmetry = str(fermion.symmetry).upper()
        if state_symmetry != symmetry:
            raise ValueError(
                "The Fermion symmetry must match the sampled Symmray MPS; "
                f"got {symmetry!r} for a {state_symmetry!r} state."
            )

        expected_dim = 4 if bool(fermion.spinful) else 2
        tables = []
        for site, code_map in enumerate(state["physical_code_maps"]):
            if tuple(code_map) != tuple(range(len(code_map))):
                raise ValueError(
                    "Symmray physical codes must be contiguous at site "
                    f"{site}; got {sorted(code_map)!r}."
                )
            if len(code_map) != expected_dim:
                raise ValueError(
                    "The sampled Symmray MPS physical dimension is incompatible "
                    f"with this {'spinful' if fermion.spinful else 'spinless'} "
                    "Fermion."
                )
            tables.append(tuple(
                _fermion_symmray_occupations(charge, offset, fermion)
                for charge, offset in code_map.values()
            ))
        return FermionConfigurationEncoding(
            symmetry=symmetry,
            spinful=bool(fermion.spinful),
            code_to_occupations=tuple(tables),
        )

    @staticmethod
    def _normalize_fermion_diagonal_observable(observable):
        aliases = {
            "n": "occupation",
            "number": "occupation",
            "occupation": "occupation",
            "density": "occupation",
            "total_charge": "total_charge",
            "total_number": "total_charge",
            "total_occupation": "total_charge",
            "doublon": "doublon",
            "double": "doublon",
            "double_occupancy": "doublon",
            "density_correlation": "density_correlation",
            "density_correlator": "density_correlation",
            "ninj": "density_correlation",
            "n_i_n_j": "density_correlation",
        }
        key = str(observable).strip().lower().replace("-", "_")
        try:
            return aliases[key]
        except KeyError as exc:
            allowed = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                "Unknown fermion diagonal observable "
                f"{observable!r}. Expected one of: {allowed}."
            ) from exc

    def _normalize_fermion_sites(self, sites):
        if sites is None:
            return tuple(range(self._L))
        if isinstance(sites, (int, np.integer)):
            sites = (int(sites),)
        else:
            try:
                sites = tuple(int(site) for site in sites)
            except TypeError as exc:
                raise TypeError("sites must be an integer or an iterable of integers.") from exc
        if not sites:
            raise ValueError("sites must contain at least one physical site.")
        if len(set(sites)) != len(sites):
            raise ValueError("sites must not contain duplicates.")
        invalid = [site for site in sites if not 0 <= site < self._L]
        if invalid:
            raise ValueError(
                f"sites contain values outside the MPS range 0..{self._L - 1}: "
                f"{invalid!r}."
            )
        return sites

    def _normalize_fermion_pairs(self, pairs):
        if pairs is None:
            raise ValueError("density_correlation requires pairs=((i, j), ...).")
        if (
            isinstance(pairs, tuple)
            and len(pairs) == 2
            and all(isinstance(site, (int, np.integer)) for site in pairs)
        ):
            pairs = (pairs,)
        try:
            pairs = tuple(tuple(int(site) for site in pair) for pair in pairs)
        except TypeError as exc:
            raise TypeError("pairs must be an (i, j) pair or iterable of pairs.") from exc
        if not pairs:
            raise ValueError("pairs must contain at least one physical pair.")
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Each density-correlation pair must have two sites.")
            if pair[0] == pair[1]:
                raise ValueError("Density-correlation pairs must contain distinct sites.")
            self._normalize_fermion_sites(pair)
        return pairs

    @staticmethod
    def _fermion_symmray_diagonal_values(charge, offset, fermion):
        """Decode standard Fermion occupations from one Symmray sector code."""
        occupations = _fermion_symmray_occupations(charge, offset, fermion)
        if not bool(fermion.spinful):
            return float(occupations[0]), 0.0
        return float(sum(occupations)), float(occupations == (1, 1))

    def _fermion_diagonal_tables(self, fermion):
        """Build physical-code lookup tables for occupation and doublon values."""
        fermion = self._resolve_fermion(fermion)

        if self._symmray_state is None:
            try:
                number = np.real(
                    np.diag(_backend_array_to_numpy(fermion.dense_operator("number")))
                ).astype(float, copy=False)
                double = (
                    np.real(
                        np.diag(_backend_array_to_numpy(fermion.dense_operator("double")))
                    ).astype(float, copy=False)
                    if bool(fermion.spinful)
                    else np.zeros_like(number, dtype=float)
                )
            except AttributeError as exc:
                raise TypeError(
                    "fermion must expose dense_operator(name) for dense MPS sampling."
                ) from exc
            return tuple((number, double) for _ in range(self._L))

        state_symmetry = str(self._symmray_state["sites"][0]["data"].symmetry)
        fermion_symmetry = str(fermion.symmetry)
        if state_symmetry != fermion_symmetry:
            raise ValueError(
                "The Fermion symmetry must match the sampled Symmray MPS; "
                f"got {fermion_symmetry!r} for a {state_symmetry!r} state."
            )
        expected_dim = 4 if bool(fermion.spinful) else 2
        tables = []
        for code_map in self._symmray_state["physical_code_maps"]:
            if len(code_map) != expected_dim:
                raise ValueError(
                    "The sampled Symmray MPS physical dimension is incompatible "
                    f"with this {'spinful' if fermion.spinful else 'spinless'} Fermion."
                )
            number = np.empty(len(code_map), dtype=float)
            double = np.empty(len(code_map), dtype=float)
            for code, (charge, offset) in code_map.items():
                number[code], double[code] = self._fermion_symmray_diagonal_values(
                    charge,
                    offset,
                    fermion,
                )
            tables.append((number, double))
        return tuple(tables)

    def fermion_diagonal_values(
        self,
        configs,
        fermion=None,
        observable=None,
        *,
        sites=None,
        pairs=None,
    ):
        """Evaluate a diagonal fermionic observable on configurations.

        This supports spinful and spinless :class:`pepsy.tensors.Fermion`
        conventions. ``"occupation"`` returns the mean local occupation on
        ``sites`` (all sites by default), ``"total_charge"`` its sum,
        ``"doublon"`` the mean ``n_up n_down`` on ``sites``, and
        ``"density_correlation"`` the mean ``n_i n_j`` across ``pairs``.
        Symmray physical codes are decoded from the source charge map, so the
        spinful Z2 even-sector ordering remains correct.
        """
        if observable is None:
            observable, fermion = fermion, None
        if observable is None:
            raise TypeError("observable is required.")
        fermion = self._resolve_fermion(fermion)
        observable = self._normalize_fermion_diagonal_observable(observable)
        configs = self._symmray_config_rows(configs, L=self._L)
        tables = self._fermion_diagonal_tables(fermion)
        occupations = np.empty(configs.shape, dtype=float)
        doublons = np.empty(configs.shape, dtype=float)
        for site, (number, double) in enumerate(tables):
            codes = configs[:, site]
            invalid = (codes < 0) | (codes >= len(number))
            if np.any(invalid):
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            occupations[:, site] = number[codes]
            doublons[:, site] = double[codes]

        if observable == "density_correlation":
            if sites is not None:
                raise ValueError("density_correlation uses pairs= rather than sites=.")
            pairs = self._normalize_fermion_pairs(pairs)
            return np.mean(
                [occupations[:, i] * occupations[:, j] for i, j in pairs],
                axis=0,
            )

        if pairs is not None:
            raise ValueError(f"{observable} does not accept pairs=.")
        sites = self._normalize_fermion_sites(sites)
        if observable == "occupation":
            return occupations[:, sites].mean(axis=1)
        if observable == "total_charge":
            return occupations[:, sites].sum(axis=1)
        if not bool(fermion.spinful):
            raise ValueError("doublon requires a spinful Fermion.")
        return doublons[:, sites].mean(axis=1)

    def estimate_fermion_diagonal(
        self,
        fermion=None,
        observable=None,
        n_samples: int = 1024,
        seed: int | None = None,
        *,
        sites=None,
        pairs=None,
    ) -> MpsDiagonalEstimate:
        """Estimate a diagonal fermionic observable from Born samples.

        The returned uncertainty uses the unbiased sample variance. This
        sampler deliberately covers diagonal observables only;
        hopping, pairing, and spin-flip observables require a fermionic local
        estimator based on amplitude ratios.
        """
        if observable is None:
            observable, fermion = fermion, None
        if observable is None:
            raise TypeError("observable is required.")
        fermion = self._resolve_fermion(fermion)
        configs, _ = self.sample_arrays(
            n_samples,
            seed=seed,
            to_numpy=True,
        )
        observable = self._normalize_fermion_diagonal_observable(observable)
        values = self.fermion_diagonal_values(
            configs,
            fermion,
            observable,
            sites=sites,
            pairs=pairs,
        )
        n_samples = int(values.size)
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(n_samples))
            if n_samples > 1
            else math.nan
        )
        return MpsDiagonalEstimate(
            mean=float(np.mean(values)),
            standard_error=standard_error,
            n_samples=n_samples,
            observable=observable,
            sites=(
                ()
                if observable == "density_correlation"
                else self._normalize_fermion_sites(sites)
            ),
            pairs=(
                self._normalize_fermion_pairs(pairs)
                if observable == "density_correlation"
                else ()
            ),
        )

    @staticmethod
    def _torch_site_ops(arrays):
        import torch  # pylint: disable=import-outside-toplevel

        device = arrays[0].device
        dtype = arrays[0].dtype
        if not (torch.is_floating_point(arrays[0]) or torch.is_complex(arrays[0])):
            dtype = torch.float64
        arrays = tuple(array.to(device=device, dtype=dtype) for array in arrays)
        right_envs = [None] * (len(arrays) + 1)
        right_envs[-1] = torch.ones((1, 1), dtype=dtype, device=device)
        for i in range(len(arrays) - 1, -1, -1):
            array = arrays[i]
            right_envs[i] = torch.einsum(
                "asb,bc,dsc->ad",
                array.conj(),
                right_envs[i + 1],
                array,
            )
        norm = right_envs[0].reshape(()).real
        if (
            (not bool(torch.isfinite(norm).detach().cpu().item()))
            or float(norm.detach().cpu().item()) <= 0.0
        ):
            raise ValueError("MPS must have a finite non-zero norm.")
        site_ops = tuple(
            (
                array.reshape(array.shape[0], array.shape[1] * array.shape[2])
                .contiguous(),
                int(array.shape[1]),
                int(array.shape[2]),
            )
            for array in arrays
        )
        return device, dtype, site_ops, tuple(right_envs), norm

    @staticmethod
    def _array_namespace_site_ops(arrays, *, backend):
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        dtype = np.dtype(getattr(arrays[0], "dtype", np.float64))
        if dtype.kind not in {"f", "c"}:
            dtype = np.dtype(np.float64)
        arrays = tuple(array.astype(dtype, copy=False) for array in arrays)
        right_envs = [None] * (len(arrays) + 1)
        right_envs[-1] = xp.ones((1, 1), dtype=dtype)
        for i in range(len(arrays) - 1, -1, -1):
            array = arrays[i]
            right_envs[i] = xp.einsum(
                "asb,bc,dsc->ad",
                xp.conjugate(array),
                right_envs[i + 1],
                array,
            )
        norm = right_envs[0].reshape(()).real
        norm_value = float(norm.get()) if backend == "cupy" else float(norm)
        if (not np.isfinite(norm_value)) or norm_value <= 0.0:
            raise ValueError("MPS must have a finite non-zero norm.")
        site_ops = tuple(
            (
                xp.ascontiguousarray(
                    array.reshape((array.shape[0], array.shape[1] * array.shape[2]))
                ),
                int(array.shape[1]),
                int(array.shape[2]),
            )
            for array in arrays
        )
        return xp, dtype, site_ops, tuple(right_envs), norm

    @staticmethod
    def _prepare_site_ops(backend, arrays):
        if backend == "torch":
            return MpsSampler._torch_site_ops(arrays)
        return MpsSampler._array_namespace_site_ops(arrays, backend=backend)

    @staticmethod
    def _torch_amplitude_site_ops(arrays):
        """Prepare amplitude contractions without retaining right environments."""
        import torch  # pylint: disable=import-outside-toplevel

        device = arrays[0].device
        dtype = arrays[0].dtype
        if not (torch.is_floating_point(arrays[0]) or torch.is_complex(arrays[0])):
            dtype = torch.float64
        arrays = tuple(array.to(device=device, dtype=dtype) for array in arrays)
        right_env = torch.ones((1, 1), dtype=dtype, device=device)
        for array in reversed(arrays):
            right_env = torch.einsum(
                "asb,bc,dsc->ad",
                array.conj(),
                right_env,
                array,
            )
        norm = right_env.reshape(()).real
        if (
            (not bool(torch.isfinite(norm).detach().cpu().item()))
            or float(norm.detach().cpu().item()) <= 0.0
        ):
            raise ValueError("MPS must have a finite non-zero norm.")
        site_ops = tuple(
            (
                array.reshape(array.shape[0], array.shape[1] * array.shape[2])
                .contiguous(),
                int(array.shape[1]),
                int(array.shape[2]),
            )
            for array in arrays
        )
        return device, dtype, site_ops, (), norm

    @staticmethod
    def _array_namespace_amplitude_site_ops(arrays, *, backend):
        """Prepare NumPy/CuPy amplitudes without retaining right environments."""
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        dtype = np.dtype(getattr(arrays[0], "dtype", np.float64))
        if dtype.kind not in {"f", "c"}:
            dtype = np.dtype(np.float64)
        arrays = tuple(array.astype(dtype, copy=False) for array in arrays)
        right_env = xp.ones((1, 1), dtype=dtype)
        for array in reversed(arrays):
            right_env = xp.einsum(
                "asb,bc,dsc->ad",
                xp.conjugate(array),
                right_env,
                array,
            )
        norm = right_env.reshape(()).real
        norm_value = float(norm.get()) if backend == "cupy" else float(norm)
        if (not np.isfinite(norm_value)) or norm_value <= 0.0:
            raise ValueError("MPS must have a finite non-zero norm.")
        site_ops = tuple(
            (
                xp.ascontiguousarray(
                    array.reshape((array.shape[0], array.shape[1] * array.shape[2]))
                ),
                int(array.shape[1]),
                int(array.shape[2]),
            )
            for array in arrays
        )
        return xp, dtype, site_ops, (), norm

    def _get_native_amplitude_site_ops(self, *, track_grad=False):
        """Return amplitude-only site data, caching only inference data."""
        if not track_grad and self._native_amplitude_site_ops is not None:
            return self.resolved_backend, self._native_amplitude_site_ops

        arrays = self._native_arrays
        if self.resolved_backend == "torch":
            if not track_grad:
                import torch  # pylint: disable=import-outside-toplevel

                arrays = tuple(array.detach() for array in arrays)
                with torch.no_grad():
                    site_data = self._torch_amplitude_site_ops(arrays)
            else:
                site_data = self._torch_amplitude_site_ops(arrays)
        else:
            site_data = self._array_namespace_amplitude_site_ops(
                arrays,
                backend=self.resolved_backend,
            )
        if not track_grad:
            self._native_amplitude_site_ops = site_data
        return self.resolved_backend, site_data

    @staticmethod
    def _torch_sample(site_data, n_samples, seed, *, to_numpy):
        import torch  # pylint: disable=import-outside-toplevel

        device, dtype, site_ops, right_envs, _norm = site_data
        vec = torch.ones((int(n_samples), 1), dtype=dtype, device=device)
        probs_total = torch.ones((int(n_samples),), dtype=torch.float64, device=device)
        batch = torch.arange(int(n_samples), device=device)
        configs = []
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        for branch_mat, phys_dim, right_dim in site_ops:
            site = len(configs)
            right_env = right_envs[site + 1]
            amps = (vec @ branch_mat).reshape(-1, phys_dim, right_dim)
            weights = (amps.conj() * (amps @ right_env)).sum(dim=2).real
            weights = weights.clamp_min(0.0)
            probs = weights / weights.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).tiny
            )
            if phys_dim == 2:
                draws = torch.rand(
                    (int(n_samples),),
                    dtype=probs.dtype,
                    device=device,
                    generator=generator,
                )
                choices = (draws >= probs[:, 0]).to(dtype=torch.long)
            else:
                choices = torch.multinomial(probs, 1, generator=generator).reshape(-1)
            selected_weights = weights[batch, choices]
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / torch.sqrt(selected_weights).clamp_min(
                torch.finfo(selected_weights.dtype).tiny
            ).reshape(-1, 1).to(dtype=dtype)
            probs_total = probs_total * selected_probs.to(dtype=torch.float64)
            configs.append(choices)

        configs = torch.stack(configs, dim=1)
        if to_numpy:
            configs = np.asarray(ar.to_numpy(configs))
            probs_total = np.asarray(ar.to_numpy(probs_total))
        return configs, probs_total

    @staticmethod
    def _array_namespace_sample(site_data, n_samples, seed, *, backend, to_numpy):
        xp, dtype, site_ops, right_envs, _norm = site_data
        vec = xp.ones((int(n_samples), 1), dtype=dtype)
        probs_total = xp.ones((int(n_samples),), dtype=np.float64)
        batch = xp.arange(int(n_samples))
        configs = []
        rng = xp.random.default_rng(seed)
        for site, (branch_mat, phys_dim, right_dim) in enumerate(site_ops):
            right_env = right_envs[site + 1]
            amps = (vec @ branch_mat).reshape((-1, phys_dim, right_dim))
            weights = xp.sum(xp.conjugate(amps) * (amps @ right_env), axis=2).real
            weights = xp.maximum(weights, 0.0)
            probs = weights / xp.maximum(
                weights.sum(axis=1, keepdims=True),
                np.finfo(float).tiny,
            )
            draws = rng.random(int(n_samples))
            if phys_dim == 2:
                # Keep NumPy's historical CDF tie behavior while preserving
                # CuPy's existing direct-Bernoulli convention.
                if backend == "cupy":
                    compare = draws >= probs[:, 0]
                else:
                    compare = draws > probs[:, 0]
                choices = compare.astype(np.int64)
            else:
                cdf = xp.cumsum(probs, axis=1)
                choices = xp.sum(draws[:, None] > cdf, axis=1).astype(np.int64)
                choices = xp.minimum(choices, probs.shape[1] - 1)
            selected_weights = weights[batch, choices]
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / xp.sqrt(
                xp.maximum(selected_weights, np.finfo(float).tiny)
            )[:, None]
            probs_total = probs_total * selected_probs
            configs.append(choices)

        configs = xp.stack(configs, axis=1)
        if to_numpy:
            configs = np.asarray(ar.to_numpy(configs))
            probs_total = np.asarray(ar.to_numpy(probs_total))
        return configs, probs_total

    @staticmethod
    def _torch_compile_supported(torch):
        """Check whether the local Torch compiler has its Python headers."""
        if not hasattr(torch, "compile"):
            return False
        import sysconfig  # pylint: disable=import-outside-toplevel
        from pathlib import Path  # pylint: disable=import-outside-toplevel

        include_dir = sysconfig.get_path("include")
        return include_dir is None or (Path(include_dir) / "Python.h").is_file()

    def _compiled_torch_sample(self, site_data, n_samples, *, track_grad):
        """Run a cached compiled Torch inference batch when available."""
        if (
            not self.torch_compile
            or track_grad
            or self._torch_compile_disabled
        ):
            return None

        import torch  # pylint: disable=import-outside-toplevel

        if not self._torch_compile_supported(torch):
            self._torch_compile_disabled = True
            return None

        key = (id(site_data), int(n_samples))
        compiled = self._torch_compiled_sample_fns.get(key)
        if compiled is None:
            def run():
                return self._torch_sample(
                    site_data,
                    n_samples,
                    None,
                    to_numpy=False,
                )

            try:
                compiled = torch.compile(
                    run,
                    fullgraph=False,
                    dynamic=False,
                    mode="reduce-overhead",
                )
                result = compiled()
            except Exception:  # pragma: no cover - compiler/version dependent
                self._torch_compile_disabled = True
                return None
            self._torch_compiled_sample_fns[key] = compiled
            return result

        try:
            return compiled()
        except Exception:  # pragma: no cover - compiler/version dependent
            self._torch_compiled_sample_fns.pop(key, None)
            self._torch_compile_disabled = True
            return None

    def _native_sample_arrays(self, n_samples, seed, *, to_numpy, track_grad):
        backend, site_data = self._get_native_site_ops(track_grad=track_grad)
        if backend == "torch":
            if seed is None and not to_numpy:
                compiled = self._compiled_torch_sample(
                    site_data,
                    n_samples,
                    track_grad=track_grad,
                )
                if compiled is not None:
                    return compiled
            return self._torch_sample(
                site_data,
                n_samples,
                seed,
                to_numpy=to_numpy,
            )
        return self._array_namespace_sample(
            site_data,
            n_samples,
            seed,
            backend=backend,
            to_numpy=to_numpy,
        )

    @staticmethod
    def _torch_configs(configs, *, device, L):
        import torch  # pylint: disable=import-outside-toplevel

        configs = torch.as_tensor(configs, dtype=torch.long, device=device)
        if configs.ndim != 2 or configs.shape[1] != int(L):
            raise ValueError(
                f"configs must have shape (batch, L={int(L)}); "
                f"got {tuple(configs.shape)}."
            )
        return configs

    @staticmethod
    def _array_namespace_configs(configs, *, backend, L):
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        configs = xp.asarray(configs, dtype=np.int64)
        if configs.ndim != 2 or configs.shape[1] != int(L):
            raise ValueError(
                f"configs must have shape (batch, L={int(L)}); "
                f"got {tuple(configs.shape)}."
            )
        return configs

    @staticmethod
    def _torch_amplitudes(site_data, configs, *, L):
        import torch  # pylint: disable=import-outside-toplevel

        device, dtype, site_ops, _right_envs, norm = site_data
        configs = MpsSampler._torch_configs(configs, device=device, L=L)
        vec = torch.ones((configs.shape[0], 1), dtype=dtype, device=device)
        batch = torch.arange(configs.shape[0], device=device)

        for site, (branch_mat, phys_dim, right_dim) in enumerate(site_ops):
            choices = configs[:, site]
            if bool(((choices < 0) | (choices >= phys_dim)).any().item()):
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            amps = (vec @ branch_mat).reshape(-1, phys_dim, right_dim)
            vec = amps[batch, choices, :]
            if vec.shape[0] != batch.shape[0]:  # pragma: no cover - sanity guard
                raise RuntimeError(
                    "Batched MPS amplitude contraction changed batch size."
                )
        scale = torch.sqrt(norm.clamp_min(torch.finfo(norm.dtype).tiny)).to(dtype=dtype)
        return vec.reshape(-1) / scale

    @staticmethod
    def _array_namespace_amplitudes(site_data, configs, *, backend, L):
        xp, dtype, site_ops, _right_envs, norm = site_data
        configs = MpsSampler._array_namespace_configs(configs, backend=backend, L=L)
        vec = xp.ones((configs.shape[0], 1), dtype=dtype)
        batch = xp.arange(configs.shape[0])

        for site, (branch_mat, phys_dim, right_dim) in enumerate(site_ops):
            choices = configs[:, site]
            invalid = (choices < 0) | (choices >= phys_dim)
            invalid = (
                bool(invalid.any().get())
                if backend == "cupy"
                else bool(invalid.any())
            )
            if invalid:
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            amps = (vec @ branch_mat).reshape((-1, phys_dim, right_dim))
            vec = amps[batch, choices, :]
        scale = xp.sqrt(xp.maximum(norm, np.finfo(float).tiny)).astype(dtype)
        return vec.reshape(-1) / scale

    @staticmethod
    def _torch_probabilities(site_data, configs, *, L):
        import torch  # pylint: disable=import-outside-toplevel

        device, dtype, site_ops, right_envs, _norm = site_data
        configs = MpsSampler._torch_configs(configs, device=device, L=L)
        vec = torch.ones((configs.shape[0], 1), dtype=dtype, device=device)
        probs_total = torch.ones(
            (configs.shape[0],),
            dtype=torch.float64,
            device=device,
        )
        batch = torch.arange(configs.shape[0], device=device)

        for site, (branch_mat, phys_dim, right_dim) in enumerate(site_ops):
            right_env = right_envs[site + 1]
            amps = (vec @ branch_mat).reshape(-1, phys_dim, right_dim)
            weights = (amps.conj() * (amps @ right_env)).sum(dim=2).real
            weights = weights.clamp_min(0.0)
            probs = weights / weights.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).tiny
            )
            choices = configs[:, site]
            if bool(((choices < 0) | (choices >= phys_dim)).any().item()):
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected_weights = weights[batch, choices]
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / torch.sqrt(selected_weights).clamp_min(
                torch.finfo(selected_weights.dtype).tiny
            ).reshape(-1, 1).to(dtype=dtype)
            probs_total = probs_total * selected_probs.to(dtype=torch.float64)
        return probs_total

    @staticmethod
    def _array_namespace_probabilities(site_data, configs, *, backend, L):
        xp, dtype, site_ops, right_envs, _norm = site_data
        configs = MpsSampler._array_namespace_configs(configs, backend=backend, L=L)
        vec = xp.ones((configs.shape[0], 1), dtype=dtype)
        probs_total = xp.ones((configs.shape[0],), dtype=np.float64)
        batch = xp.arange(configs.shape[0])

        for site, (branch_mat, phys_dim, right_dim) in enumerate(site_ops):
            right_env = right_envs[site + 1]
            amps = (vec @ branch_mat).reshape((-1, phys_dim, right_dim))
            weights = xp.sum(xp.conjugate(amps) * (amps @ right_env), axis=2).real
            weights = xp.maximum(weights, 0.0)
            probs = weights / xp.maximum(
                weights.sum(axis=1, keepdims=True),
                np.finfo(float).tiny,
            )
            choices = configs[:, site]
            invalid = (choices < 0) | (choices >= phys_dim)
            invalid = (
                bool(invalid.any().get())
                if backend == "cupy"
                else bool(invalid.any())
            )
            if invalid:
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected_weights = weights[batch, choices]
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / xp.sqrt(
                xp.maximum(selected_weights, np.finfo(float).tiny)
            )[:, None]
            probs_total = probs_total * selected_probs
        return probs_total

    @staticmethod
    def _to_numpy_backend_array(array, backend):
        return _backend_array_to_numpy(array)

    def amplitudes(
        self,
        configs,
        *,
        to_numpy: bool = True,
        track_grad: bool = False,
    ):
        """Return batched MPS amplitudes for ``configs``.

        ``configs`` should have shape ``(batch, L)``. Dense NumPy, Torch, and
        CuPy MPS tensors are contracted in one batched backend-native pass.
        Symmray MPSs use block-sparse contractions on the underlying NumPy,
        Torch, or CuPy backend. Set ``to_numpy=False`` to keep Torch/CuPy
        outputs on their device. Measurement calls default to inference mode;
        pass ``track_grad=True`` to retain a Torch autograd graph.
        """
        if self._symmray_state is not None:
            out = self._symmray_amplitudes(configs)
            backend = self._symmray_state["array_backend"]
            return self._to_numpy_backend_array(out, backend) if to_numpy else out

        if self._native_arrays is None:
            backend, site_data = self._get_evaluation_site_ops()
        else:
            backend, site_data = self._get_native_amplitude_site_ops(
                track_grad=bool(track_grad)
            )
        if backend == "torch":
            if track_grad:
                out = self._torch_amplitudes(site_data, configs, L=self._L)
            else:
                import torch  # pylint: disable=import-outside-toplevel

                with torch.no_grad():
                    out = self._torch_amplitudes(site_data, configs, L=self._L)
        else:
            out = self._array_namespace_amplitudes(
                site_data,
                configs,
                backend=backend,
                L=self._L,
            )
        return self._to_numpy_backend_array(out, backend) if to_numpy else out

    def single_site_flip_amplitude_ratios(
        self,
        configs,
        *,
        to_numpy: bool = True,
        block_size: int = 16,
    ):
        """Return ``psi(x with site flipped) / psi(x)`` for binary MPSs.

        This dense-native helper is intended for local-energy estimators. It
        evaluates all one-site flips with prefix/suffix contractions instead
        of making one complete MPS amplitude sweep per flipped configuration.
        The returned array has shape ``(batch, L)``. Native NumPy, Torch, and
        CuPy MPSs are supported; Symmray and legacy Quimb samplers should use
        :meth:`amplitudes` on explicitly connected configurations instead.

        ``block_size`` limits the prefix workspace retained while forming the
        ratios. Smaller values reduce peak memory at the cost of a little more
        backend work.
        """
        if not isinstance(block_size, Integral) or int(block_size) < 1:
            raise ValueError("block_size must be a positive integer.")
        if self._symmray_state is not None or self._native_arrays is None:
            raise NotImplementedError(
                "single-site flip ratios require a dense native MPS sampler."
            )

        backend, _site_data = self._get_native_amplitude_site_ops(
            track_grad=False
        )
        arrays = self._native_arrays
        L = len(arrays)
        if any(int(array.shape[1]) != 2 for array in arrays):
            raise NotImplementedError(
                "single-site flip ratios require binary physical dimensions."
            )
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            with torch.no_grad():
                ratios = self._single_site_flip_ratios_native(
                    arrays,
                    configs,
                    backend=backend,
                    block_size=int(block_size),
                    L=L,
                )
        else:
            ratios = self._single_site_flip_ratios_native(
                arrays,
                configs,
                backend=backend,
                block_size=int(block_size),
                L=L,
            )
        return self._to_numpy_backend_array(ratios, backend) if to_numpy else ratios

    @staticmethod
    def _single_site_flip_ratios_native(
        arrays,
        configs,
        *,
        backend,
        block_size,
        L,
    ):
        """Form dense-native single-site flip ratios with bounded workspace."""
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            device = arrays[0].device
            dtype = arrays[0].dtype
            configs = MpsSampler._torch_configs(configs, device=device, L=L)
            xp = torch
        else:
            xp = np
            if backend == "cupy":
                import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

            configs = MpsSampler._array_namespace_configs(
                configs,
                backend=backend,
                L=L,
            )
            dtype = arrays[0].dtype

        invalid = (configs < 0) | (configs > 1)
        if backend == "torch":
            invalid = bool(invalid.any().item())
        elif backend == "cupy":
            invalid = bool(invalid.any().get())
        else:
            invalid = bool(invalid.any())
        if invalid:
            raise ValueError("configs contain non-binary physical indices.")

        n_samples = int(configs.shape[0])
        block_size = min(int(block_size), L)
        block_starts = tuple(range(0, L, block_size))

        def contract_left(left, choices, array):
            mask_zero = choices == 0
            mask_one = ~mask_zero
            if backend == "torch":
                output = xp.empty(
                    (n_samples, int(array.shape[2])),
                    dtype=dtype,
                    device=device,
                )
            else:
                output = xp.empty(
                    (n_samples, int(array.shape[2])),
                    dtype=dtype,
                )
            if backend == "torch":
                output[mask_zero] = left[mask_zero] @ array[:, 0, :]
                output[mask_one] = left[mask_one] @ array[:, 1, :]
            else:
                output[mask_zero] = xp.einsum(
                    "nl,lr->nr", left[mask_zero], array[:, 0, :]
                )
                output[mask_one] = xp.einsum(
                    "nl,lr->nr", left[mask_one], array[:, 1, :]
                )
            return output

        def contract_right(choices, array, right):
            mask_zero = choices == 0
            mask_one = ~mask_zero
            if backend == "torch":
                output = xp.empty(
                    (n_samples, int(array.shape[0])),
                    dtype=dtype,
                    device=device,
                )
            else:
                output = xp.empty(
                    (n_samples, int(array.shape[0])),
                    dtype=dtype,
                )
            if backend == "torch":
                output[mask_zero] = right[mask_zero] @ array[:, 0, :].transpose(0, 1)
                output[mask_one] = right[mask_one] @ array[:, 1, :].transpose(0, 1)
            else:
                output[mask_zero] = xp.einsum(
                    "lr,nr->nl", array[:, 0, :], right[mask_zero]
                )
                output[mask_one] = xp.einsum(
                    "lr,nr->nl", array[:, 1, :], right[mask_one]
                )
            return output

        def contract_site(left, array, choices, right):
            if backend == "torch":
                values = contract_left(left, choices, array)
                return (values * right).sum(dim=1)
            values = contract_left(left, choices, array)
            return xp.sum(values * right, axis=1)

        def ones(width):
            if backend == "torch":
                return xp.ones((n_samples, width), dtype=dtype, device=configs.device)
            return xp.ones((n_samples, width), dtype=dtype)

        # The parent amplitude is shared by every local ratio and is therefore
        # evaluated once, independently of the block workspace below.
        prefix = ones(1)
        for site in range(L):
            choices = configs[:, site]
            prefix = contract_left(
                prefix,
                choices,
                arrays[site],
            )
        parent = prefix[:, 0]

        # Save only suffixes at block boundaries. Prefixes inside one block
        # are retained briefly, keeping peak workspace independent of L.
        suffix_boundaries = {L: ones(1)}
        suffix = suffix_boundaries[L]
        block_start_set = set(block_starts[1:])
        for site in range(L - 1, -1, -1):
            choices = configs[:, site]
            suffix = contract_right(
                choices,
                arrays[site],
                suffix,
            )
            if site in block_start_set:
                suffix_boundaries[site] = suffix

        prefix = ones(1)
        if backend == "torch":
            ratios = xp.empty((n_samples, L), dtype=dtype, device=device)
        else:
            ratios = xp.empty((n_samples, L), dtype=dtype)
        for start in block_starts:
            stop = min(start + block_size, L)
            block_prefixes = [prefix]
            for site in range(start, stop):
                choices = configs[:, site]
                prefix = contract_left(
                    prefix,
                    choices,
                    arrays[site],
                )
                block_prefixes.append(prefix)

            suffix = suffix_boundaries[stop]
            for offset in range(stop - start - 1, -1, -1):
                site = start + offset
                choices = configs[:, site]
                flipped = 1 - choices
                numerator = contract_site(
                    block_prefixes[offset],
                    arrays[site],
                    flipped,
                    suffix,
                )
                ratios[:, site] = numerator / parent
                suffix = contract_right(
                    choices,
                    arrays[site],
                    suffix,
                )

        return ratios

    def probabilities(self, configs, *, to_numpy: bool = True):
        """Return normalized Born probabilities for batched ``configs``.

        This follows the same conditional-probability sweep as sampling, but
        with user-supplied physical indices. Dense MPSs avoid looping over
        configurations and run on Torch/CuPy when the tensors do. Symmray MPSs
        use charge-aware block-sparse conditionals without densifying state
        tensors.
        """
        if self._symmray_state is not None:
            out = self._symmray_probabilities(configs)
            backend = self._symmray_state["array_backend"]
            return self._to_numpy_backend_array(out, backend) if to_numpy else out

        backend, site_data = self._get_evaluation_site_ops()
        if backend == "torch":
            out = self._torch_probabilities(site_data, configs, L=self._L)
        else:
            out = self._array_namespace_probabilities(
                site_data,
                configs,
                backend=backend,
                L=self._L,
            )
        return self._to_numpy_backend_array(out, backend) if to_numpy else out

    def sample_arrays(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        to_numpy: bool = False,
        track_grad: bool = False,
    ):
        """Draw samples and return raw ``(configs, probs)`` arrays.

        With ``backend="native"`` or ``backend="symmray"``, this returns
        backend-native arrays by default: Torch tensors stay on Torch and CuPy
        arrays stay on CuPy. Symmray inputs retain block-sparse state tensors;
        only the returned discrete configurations and probabilities are dense.
        The returned ``configs`` have shape ``(n_samples, L)`` and ``probs``
        has shape ``(n_samples,)``. Set ``to_numpy=True`` to force CPU NumPy
        arrays, matching the legacy :meth:`sample` result conversion.
        Sampling is inference-only by default; set ``track_grad=True`` to
        retain a Torch autograd graph for the sampled Born probabilities.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(track_grad, (bool, np.bool_)):
            raise TypeError("track_grad must be a boolean.")
        if self._symmray_state is not None:
            def sample_symmray():
                strategy, selection, estimated_bytes = (
                    self._resolve_symmray_sampling_strategy(int(n_samples))
                )
                configs, probs, stats = self._symmray_sample_arrays(
                    self._symmray_state,
                    int(n_samples),
                    seed,
                    strategy=strategy,
                    max_prefix_groups=self.max_prefix_groups,
                    to_numpy=to_numpy,
                )
                stats.update(
                    {
                        "requested_strategy": self.prefix_strategy,
                        "strategy_selection": selection,
                        "estimated_dense_site_bytes": (
                            None
                            if estimated_bytes is None
                            else int(estimated_bytes)
                        ),
                        "dense_memory_limit_bytes": self.dense_memory_limit,
                    }
                )
                return configs, probs, stats

            if (
                self._symmray_state["array_backend"] == "torch"
                and not track_grad
            ):
                import torch  # pylint: disable=import-outside-toplevel

                with torch.no_grad():
                    configs, probs, stats = sample_symmray()
                self._last_symmray_sampling_stats = stats
                return configs, probs
            configs, probs, stats = sample_symmray()
            self._last_symmray_sampling_stats = stats
            return configs, probs
        if self._native_arrays is not None:
            return self._native_sample_arrays(
                int(n_samples),
                seed,
                to_numpy=to_numpy,
                track_grad=bool(track_grad),
            )

        configs = []
        probs = []
        for config, prob in self._psi.sample(int(n_samples), seed=seed):
            configs.append(config)
            probs.append(prob)
        return np.asarray(configs, dtype=np.int64), np.asarray(probs, dtype=float)

    def sample_batch(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        to_numpy: bool = False,
        track_grad: bool = False,
        fermion=None,
    ) -> MpsBatchSampleResult:
        """Draw samples and return a named batched result.

        This is the preferred API for fast downstream workflows. With
        ``backend="native"`` and ``to_numpy=False``, Torch/CuPy arrays stay on
        their current device. Use :meth:`sample_arrays` when tuple unpacking is
        more convenient, or :meth:`sample` when the legacy Python-list/grid
        result is needed. Pass ``fermion=...`` for a Symmray fermionic MPS to
        attach a :class:`FermionConfigurationEncoding`; downstream code can
        then call :meth:`MpsBatchSampleResult.occupations` without guessing the
        physical-code convention.
        """
        configs, probs = self.sample_arrays(
            n_samples,
            seed=seed,
            to_numpy=to_numpy,
            track_grad=track_grad,
        )
        backend = (
            "numpy"
            if to_numpy
            else (
                self._symmray_state["array_backend"]
                if self._symmray_state is not None
                else (self.resolved_backend if self._native_arrays is not None else "numpy")
            )
        )
        return MpsBatchSampleResult(
            configs=configs,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
            one_d_to_two_d=dict(self.one_d_to_two_d),
            backend=backend,
            configuration_encoding=(
                self.fermion_configuration_encoding(
                    self.fermion if fermion is None else fermion
                )
                if (self.fermion is not None or fermion is not None)
                else None
            ),
        )

    def sample(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        track_grad: bool = False,
    ) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the MPS.

        The dense native backend uses batched conditional contractions on the
        MPS tensor device. The Symmray backend caches a right-canonical copy
        and sweeps block-sparse physical slices left-to-right. The quimb
        backend uses ``MatrixProductState.sample()``, which internally
        right-canonicalizes the MPS and sweeps left-to-right.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        return self.sample_batch(
            n_samples,
            seed=seed,
            to_numpy=True,
            track_grad=track_grad,
        ).to_sample_result()


class VecSampler:
    """Sample from a dense state vector (e.g. from MpsOptimizer mode='exact').

    Computes Born probabilities ``p_i = |ψ_i|²`` and samples configurations
    from the resulting categorical distribution. NumPy, Torch, and CuPy
    state vectors remain on their original backend for probability evaluation
    and batched sampling. Use :meth:`MpsBatchSampleResult.to_numpy` or the
    legacy :meth:`sample` method when a host-side result is explicitly wanted.

    Parameters
    ----------
    state : TensorNetwork, Tensor, or array-like
        The dense state. If a quimb TensorNetwork/Tensor, the physical indices
        are assumed to follow ``ind_id`` format (default ``'k{}'``). If a raw
        array, it is reshaped to a 1D vector of length 2^L.
    one_d_to_two_d : dict[int, tuple[int, int]], optional
        Mapping from 1D site index to (x, y) lattice coordinate. If omitted,
        infer a trivial single-row map from the dense vector length or state
        object's ``L`` attribute.
    ind_id : str
        Format string for physical index names (default ``'k{}'``).
    basis : str or sequence[str], optional
        Sampling basis is supplied to :meth:`sample`, rather than fixed at
        construction. It can be global ``"X"``, ``"Y"``, or ``"Z"``, the
        string ``"random"`` for an independent random choice at each site,
        or a length-``L`` sequence such as ``"XYZZZ"``.
    """

    def __init__(
        self,
        state,
        one_d_to_two_d: dict[int, tuple[int, int]] | None = None,
        ind_id: str = "k{}",
    ):
        if one_d_to_two_d is None:
            inferred_L = getattr(state, "L", None)
            if inferred_L is None:
                inferred_L = self._infer_state_length(state)
            one_d_to_two_d = {
                site: (site, 0) for site in range(int(inferred_L))
            }
        L = _validate_one_d_to_two_d(one_d_to_two_d)
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1
        self._L = L
        self._ind_id = ind_id
        self._state = None
        self._vector = None
        self._probs = None
        self.backend = None
        self.resolved_backend = None
        self._basis_probability_cache = {}
        self._basis_probability_cache_limit = 16
        self.refresh(state)

    @property
    def L(self) -> int:
        """Number of sites represented by the dense state vector."""
        return self._L

    def refresh(self, state=None):
        """Refresh the cached vector and basis distributions from ``state``.

        This mirrors :meth:`MpsSampler.refresh`: call it after replacing or
        evolving the dense state vector. The site map and resolved backend are
        fixed by the sampler's construction.
        """
        if state is None:
            state = self._state
        if state is None:
            raise ValueError("refresh requires a dense state vector.")

        # Extract the state vector with correct index ordering. Keep the
        # native array backend: exact GPU statevectors must not be copied to
        # NumPy merely to construct the sampler.
        vec = self._to_vector(state, self._L, self._ind_id)
        backend = _mps_array_backend(vec)
        if backend == "symmray" and hasattr(vec, "to_dense"):
            vec = vec.to_dense()
            backend = _mps_array_backend(vec)
        if backend not in {"numpy", "torch", "cupy"}:
            vec = np.asarray(vec)
            backend = "numpy"

        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.is_complex(vec):
                target_dtype = (
                    torch.complex128
                    if vec.dtype in {torch.float64, torch.int64}
                    else torch.complex64
                )
                vec = vec.to(dtype=target_dtype)
        elif backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            if not cp.iscomplexobj(vec):
                target_dtype = cp.complex128 if vec.dtype == cp.float64 else cp.complex64
                vec = vec.astype(target_dtype, copy=False)
        else:
            vec = np.asarray(vec)
            if not np.iscomplexobj(vec):
                target_dtype = np.complex128 if vec.dtype == np.float64 else np.complex64
                vec = vec.astype(target_dtype, copy=False)
            elif vec.dtype not in {np.dtype("complex64"), np.dtype("complex128")}:
                vec = vec.astype(np.complex128, copy=False)

        vec = vec.reshape(-1)
        expected_size = 2 ** self._L
        vec_size = int(vec.numel()) if backend == "torch" else int(vec.size)
        if vec_size != expected_size:
            raise ValueError(
                f"state vector size must be 2**L={expected_size} for L={self._L}; "
                f"got {vec_size}."
            )

        # Compute and cache Born probabilities on the state backend. The
        # scalar norm check necessarily inspects one scalar on the host, but
        # the vector and all probability arrays remain device resident.
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            probs = torch.abs(vec) ** 2
            total = probs.sum()
            total_value = float(total.detach().cpu())
        elif backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            probs = cp.abs(vec) ** 2
            total = probs.sum()
            total_value = float(cp.asnumpy(total))
        else:
            probs = np.abs(vec) ** 2
            total = probs.sum()
            total_value = float(total)
        if not np.isfinite(total_value) or total_value <= 0:
            raise ValueError("state vector must have a finite non-zero norm.")
        probs /= total
        if backend == "torch":
            normalized_vector = vec / torch.sqrt(total)
        elif backend == "cupy":
            normalized_vector = vec / cp.sqrt(total)
        else:
            normalized_vector = vec / np.sqrt(total)
        self._vector = normalized_vector
        self._probs = probs
        self.backend = backend
        self.resolved_backend = backend
        self._state = state
        self._basis_probability_cache = {("Z",) * self._L: self._probs}
        self._basis_probability_cache_limit = 16
        return self

    @staticmethod
    def _infer_state_length(state):
        """Infer ``L`` from an MPS-like object or raw 2**L vector."""
        if hasattr(state, "inds"):
            physical_inds = [
                ind for ind in state.inds if str(ind).startswith("k")
            ]
            if physical_inds:
                return len(physical_inds)
        data = getattr(state, "data", None)
        if data is None or not hasattr(data, "shape"):
            data = state
        if hasattr(data, "numel"):
            size = int(data.numel())
        elif hasattr(data, "size"):
            size_attr = data.size
            size = int(size_attr() if callable(size_attr) else size_attr)
        else:
            size = int(np.asarray(data).size)
        if size < 1 or size & (size - 1):
            raise ValueError(
                "Cannot infer a site count: dense state length must be a "
                "positive power of two."
            )
        return int(size.bit_length() - 1)

    @staticmethod
    def _to_vector(state, L, ind_id):
        """Convert state to a flat vector with sites in 0..L-1 order."""
        import quimb.tensor as qtn  # noqa: F811

        if isinstance(state, qtn.TensorNetwork):
            # Contract to a single tensor with ordered physical indices
            inds = [ind_id.format(i) for i in range(L)]
            t = state.contract(all, output_inds=inds)
            return t.data.reshape(-1)
        if isinstance(state, qtn.Tensor):
            inds = [ind_id.format(i) for i in range(L)]
            return state.transpose(*inds).data.reshape(-1)
        # Raw array
        if hasattr(state, "reshape"):
            return state.reshape(-1)
        return np.asarray(state).reshape(-1)

    def _config_indices(self, configs):
        """Validate binary configs and return backend-native basis indices."""
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            if isinstance(configs, torch.Tensor):
                values = configs.to(device=self._vector.device, dtype=torch.long)
            else:
                values = torch.as_tensor(
                    configs,
                    device=self._vector.device,
                    dtype=torch.long,
                )
            invalid = bool(((values < 0) | (values > 1)).any().item())
        elif self.backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            values = cp.asarray(configs, dtype=cp.int64)
            invalid = bool(cp.any((values < 0) | (values > 1)).item())
        else:
            values = np.asarray(configs, dtype=np.int64)
            invalid = bool(np.any((values < 0) | (values > 1)))
        if values.ndim != 2 or values.shape[1] != self._L:
            raise ValueError(
                f"configs must have shape (batch, L={self._L}); got {values.shape}."
            )
        if invalid:
            raise ValueError("configs contain non-binary physical indices.")

        if self.backend == "torch":
            powers = 2 ** torch.arange(
                self._L - 1,
                -1,
                -1,
                dtype=torch.long,
                device=values.device,
            )
        elif self.backend == "cupy":
            powers = 2 ** cp.arange(self._L - 1, -1, -1, dtype=cp.int64)
        else:
            powers = 2 ** np.arange(self._L - 1, -1, -1, dtype=np.int64)
        return values, values @ powers

    def amplitudes(
        self,
        configs,
        *,
        to_numpy: bool = True,
        track_grad: bool = False,
    ):
        """Return dense-state amplitudes for computational-basis configs.

        The signature follows :meth:`MpsSampler.amplitudes`. ``track_grad``
        is accepted for interface compatibility; Torch indexing preserves the
        state graph when it is requested.
        """
        if not isinstance(track_grad, (bool, np.bool_)):
            raise TypeError("track_grad must be a boolean.")
        _values, indices = self._config_indices(configs)
        if self.backend == "torch" and not track_grad:
            import torch  # pylint: disable=import-outside-toplevel

            with torch.no_grad():
                amplitudes = self._vector[indices]
        else:
            amplitudes = self._vector[indices]
        return _backend_array_to_numpy(amplitudes) if to_numpy else amplitudes

    def probabilities(
        self,
        configs,
        *,
        to_numpy: bool = True,
        basis="Z",
    ):
        """Return normalized Born probabilities for batched configurations.

        ``basis`` extends the MPS-compatible API to the exact sampler's Pauli
        basis support. Configurations are interpreted as computational-basis
        outcomes in the selected basis.
        """
        _values, indices = self._config_indices(configs)
        resolved_basis = _resolve_measurement_basis(
            basis,
            self._L,
            rng=np.random.default_rng(0),
        )
        probabilities = (
            self._probs
            if resolved_basis == ("Z",) * self._L
            else self._probabilities_for_basis(resolved_basis)
        )
        probabilities = probabilities[indices]
        return _backend_array_to_numpy(probabilities) if to_numpy else probabilities

    def sample_arrays(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        to_numpy: bool = False,
        track_grad: bool = False,
        basis="Z",
        chunk_size: int | None = _DEFAULT_VEC_SAMPLE_CHUNK_SIZE,
    ):
        """Draw samples and return raw ``(configs, probs)`` arrays."""
        batch = self.sample_batch(
            n_samples=n_samples,
            seed=seed,
            to_numpy=to_numpy,
            track_grad=track_grad,
            basis=basis,
            chunk_size=chunk_size,
        )
        return batch.configs, batch.probs

    def _probabilities_for_basis(self, basis):
        """Return the normalized distribution in a resolved local basis."""
        basis = tuple(basis)
        cached = self._basis_probability_cache.get(basis)
        if cached is not None:
            return cached

        vector = self._vector.reshape((2,) * self._L)
        for site, label in enumerate(basis):
            rotation = _measurement_rotation(label, like=vector)
            if self.backend == "torch":
                import torch  # pylint: disable=import-outside-toplevel

                vector = torch.tensordot(
                    rotation,
                    vector,
                    dims=([1], [site]),
                )
                vector = torch.movedim(vector, 0, site)
            elif self.backend == "cupy":
                import cupy as cp  # pylint: disable=import-outside-toplevel

                vector = cp.tensordot(
                    rotation,
                    vector,
                    axes=(1, site),
                )
                vector = cp.moveaxis(vector, 0, site)
            else:
                vector = np.tensordot(
                    rotation,
                    vector,
                    axes=(1, site),
                )
                vector = np.moveaxis(vector, 0, site)
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            probabilities = torch.abs(vector.reshape(-1)) ** 2
        elif self.backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            probabilities = cp.abs(vector.reshape(-1)) ** 2
        else:
            probabilities = np.abs(vector.reshape(-1)) ** 2
        probabilities = probabilities / probabilities.sum()
        if len(self._basis_probability_cache) >= self._basis_probability_cache_limit:
            z_basis = ("Z",) * self._L
            oldest = next(
                key for key in self._basis_probability_cache if key != z_basis
            )
            del self._basis_probability_cache[oldest]
        self._basis_probability_cache[basis] = probabilities
        return probabilities

    def probability_vector(
        self,
        basis="Z",
        *,
        seed: int | None = None,
        to_numpy: bool = False,
    ) -> Any:
        """Return the exact normalized distribution in a measurement basis.

        ``basis="random"`` chooses one independent X/Y/Z label per site and
        uses ``seed`` to make that pattern reproducible. The returned vector
        uses the same big-endian bit ordering as computational-basis sampling.
        """
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        basis_rng = np.random.default_rng(seed)
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=basis_rng)
        probabilities = self._probabilities_for_basis(resolved_basis)
        if self.backend == "torch":
            probabilities = probabilities.clone()
        else:
            probabilities = probabilities.copy()
        return _backend_array_to_numpy(probabilities) if to_numpy else probabilities

    def _new_rng(self, seed):
        """Construct a backend-native random generator for shot draws."""
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            generator = torch.Generator(device=self._vector.device)
            if seed is not None:
                generator.manual_seed(int(seed))
            return generator
        if self.backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            return cp.random.RandomState(seed)
        return np.random.default_rng(seed)

    def sample(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        track_grad: bool = False,
        basis="Z",
        chunk_size: int | None = _DEFAULT_VEC_SAMPLE_CHUNK_SIZE,
    ) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the state vector.

        Parameters
        ----------
        basis : {"X", "Y", "Z", "random"} or sequence[str]
            Measurement basis. A sequence supplies one Pauli basis per site.
            For ``"random"``, one basis pattern is drawn per call and shared
            by all samples in that call. This is the efficient setting for
            randomized-basis batches; it is not a different basis per shot.
        chunk_size : int or None
            Number of shots generated at a time. The default uses bounded
            internal draw memory. This does not reduce the memory of the
            returned legacy lists/grids; use :meth:`iter_samples` for true
            streaming.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        return self.sample_batch(
            n_samples=n_samples,
            seed=seed,
            track_grad=track_grad,
            basis=basis,
            chunk_size=chunk_size,
        ).to_sample_result()

    def _draw_samples(
        self,
        n_samples=1,
        *,
        seed=None,
        rng=None,
        basis="Z",
        resolved_basis=None,
        probabilities=None,
    ):
        """Draw a backend-native batch without constructing legacy grids."""
        n_samples = _validate_sample_count(n_samples)
        if rng is None:
            rng = self._new_rng(seed)
        if resolved_basis is None:
            basis_rng = np.random.default_rng(seed)
            resolved_basis = _resolve_measurement_basis(
                basis,
                self._L,
                rng=basis_rng,
            )
        if probabilities is None:
            probabilities = self._probabilities_for_basis(resolved_basis)
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            # CUDA ``torch.multinomial`` currently rejects categorical
            # distributions with more than 2**24 categories.  Dense exact
            # statevectors reach 2**25 already at 25 qubits, so use inverse
            # CDF sampling for the large-vector case.  The state vector and
            # probability distribution remain backend-native.
            if probabilities.numel() > _TORCH_MULTINOMIAL_MAX_CATEGORIES:
                cdf = torch.cumsum(probabilities, dim=0)
                cdf[-1] = 1.0
                draws = torch.rand(
                    n_samples,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                    generator=rng,
                )
                indices = torch.searchsorted(cdf, draws, right=True)
            else:
                indices = torch.multinomial(
                    probabilities,
                    n_samples,
                    replacement=True,
                    generator=rng,
                )
            bit_positions = torch.arange(
                self._L - 1,
                -1,
                -1,
                dtype=torch.long,
                device=indices.device,
            )
            configs = ((indices[:, None] >> bit_positions) & 1).to(torch.int8)
            probs = probabilities[indices]
        elif self.backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            indices = cp.asarray(
                rng.choice(len(probabilities), size=n_samples, p=probabilities),
                dtype=cp.int64,
            )
            bit_positions = cp.arange(self._L - 1, -1, -1, dtype=cp.int64)
            configs = ((indices[:, None] >> bit_positions) & 1).astype(
                cp.int8,
                copy=False,
            )
            probs = probabilities[indices]
        else:
            indices = rng.choice(len(probabilities), size=n_samples, p=probabilities)
            bit_positions = np.arange(self._L - 1, -1, -1, dtype=np.int64)
            configs = ((np.asarray(indices, dtype=np.int64)[:, None] >> bit_positions) & 1).astype(
                np.int8,
                copy=False,
            )
            probs = np.asarray(probabilities[indices], dtype=float)
        basis_probability = _basis_selection_probability(basis, self._L)
        weights = probs * basis_probability
        return configs, probs, weights, resolved_basis, basis_probability

    def iter_samples(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        basis="Z",
        chunk_size: int = _DEFAULT_VEC_SAMPLE_CHUNK_SIZE,
    ):
        """Stream sampled batches with bounded shot-memory.

        The state vector and one resolved-basis probability vector remain
        resident, but only ``chunk_size`` configurations, probabilities, and
        weights are materialized at once. In ``basis="random"`` mode, one
        independent X/Y/Z basis label is chosen per site and shared by all
        yielded chunks from this call.
        """
        n_samples = _validate_sample_count(n_samples)
        chunk_size = _validate_sample_chunk_size(chunk_size)
        if chunk_size is None:
            chunk_size = max(1, n_samples)
        basis_rng = np.random.default_rng(seed)
        rng = self._new_rng(seed)
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=basis_rng)
        probabilities = self._probabilities_for_basis(resolved_basis)
        basis_probability = _basis_selection_probability(basis, self._L)
        for start in range(0, n_samples, chunk_size):
            count = min(chunk_size, n_samples - start)
            configs, probs, weights, _, _ = self._draw_samples(
                count,
                rng=rng,
                basis=basis,
                resolved_basis=resolved_basis,
                probabilities=probabilities,
            )
            yield MpsBatchSampleResult(
                configs=configs,
                probs=probs,
                Lx=self.Lx,
                Ly=self.Ly,
                one_d_to_two_d=dict(self.one_d_to_two_d),
                backend=self.backend,
                basis=resolved_basis,
                basis_probability=basis_probability,
                weights=weights,
            )

    def sample_batch(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        to_numpy: bool = False,
        track_grad: bool = False,
        basis="Z",
        chunk_size: int | None = _DEFAULT_VEC_SAMPLE_CHUNK_SIZE,
    ) -> MpsBatchSampleResult:
        """Draw a backend-native batch without Python per-shot loops.

        ``sample`` remains the compatibility API that also builds one 2D grid
        per shot. ``chunk_size`` bounds temporary draw allocations; the final
        returned arrays still contain all requested shots. Use
        :meth:`iter_samples` when the consumer can process streaming chunks.
        """
        n_samples = _validate_sample_count(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        if not isinstance(track_grad, (bool, np.bool_)):
            raise TypeError("track_grad must be a boolean.")
        chunk_size = _validate_sample_chunk_size(chunk_size)
        if chunk_size is None:
            chunk_size = max(1, n_samples)

        basis_rng = np.random.default_rng(seed)
        rng = self._new_rng(seed)
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=basis_rng)
        probabilities = self._probabilities_for_basis(resolved_basis)
        basis_probability = _basis_selection_probability(basis, self._L)
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            configs = torch.empty(
                (n_samples, self._L),
                dtype=torch.int8,
                device=self._vector.device,
            )
            probs = torch.empty(
                n_samples,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            weights = torch.empty_like(probs)
        elif self.backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            configs = cp.empty((n_samples, self._L), dtype=cp.int8)
            probs = cp.empty(n_samples, dtype=probabilities.dtype)
            weights = cp.empty_like(probs)
        else:
            configs = np.empty((n_samples, self._L), dtype=np.int8)
            probs = np.empty(n_samples, dtype=float)
            weights = np.empty(n_samples, dtype=float)
        for start in range(0, n_samples, chunk_size):
            stop = min(n_samples, start + chunk_size)
            chunk_configs, chunk_probs, chunk_weights, _, _ = self._draw_samples(
                stop - start,
                rng=rng,
                basis=basis,
                resolved_basis=resolved_basis,
                probabilities=probabilities,
            )
            configs[start:stop] = chunk_configs
            probs[start:stop] = chunk_probs
            weights[start:stop] = chunk_weights
        batch = MpsBatchSampleResult(
            configs=configs,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
            one_d_to_two_d=dict(self.one_d_to_two_d),
            backend=self.backend,
            basis=resolved_basis,
            basis_probability=basis_probability,
            weights=weights,
        )
        if self.backend == "torch" and not track_grad:
            batch.probs = batch.probs.detach()
            batch.weights = batch.weights.detach()
        return batch.to_numpy() if to_numpy else batch


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
    fit_n_iter : int, default=2
        Number of local FIT sweeps used by the ``"fit"`` ket compressor and
        by the ``"dmrg"`` future-environment preparation.
    contraction_opt : object, optional
        Quimb contraction optimizer passed to ``TensorNetwork.contract``.
        The default ``"auto-hq"`` uses the high-quality Cotengra path when
        available.
    row_cache_max_bytes : int, default=0
        Zero selects the simple conditioned-boundary sweep. A positive budget
        opts into estimated dense row-cache storage and workspace across live
        prefix groups. Oversized caches use the reference contraction. This
        is not a total process-memory cap.
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
        fit_n_iter=2,
        contraction_opt="auto-hq",
        row_cache_max_bytes=0,
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
        if (
            isinstance(fit_n_iter, (bool, np.bool_))
            or not isinstance(fit_n_iter, (int, np.integer))
            or int(fit_n_iter) < 1
        ):
            raise ValueError("fit_n_iter must be a positive integer.")
        self.fit_n_iter = int(fit_n_iter)
        self.contraction_opt = contraction_opt
        if (
            isinstance(row_cache_max_bytes, bool)
            or not isinstance(row_cache_max_bytes, (int, np.integer))
            or row_cache_max_bytes < 0
        ):
            raise ValueError("row_cache_max_bytes must be a non-negative integer.")
        self.row_cache_max_bytes = int(row_cache_max_bytes)
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
        if self.boundary_engine != "exact" and (
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
        self._row_bond_cache = {}
        self._identity_future_cache = {}
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
            )
            # Quimb owns this boundary sweep and returns the native ``ymax``
            # MPS objects. Keep the cache separate from the shot-conditioned
            # ket boundary because this network still sums over future rows.
            store.start_sweep(self._norm, "y", update_side="left")
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

    def _scaled(self, value):
        """Represent a real or complex value as mantissa times 10**exponent."""
        value = self._scalar(value)
        magnitude = abs(value)
        if magnitude == 0:
            return 0.0 if not np.iscomplexobj(value) else 0.0j, 0
        exponent = math.floor(math.log10(magnitude))
        mantissa = value / (10.0 ** exponent)
        return mantissa, exponent

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
        if previous is not None:
            max_hermiticity = real_xp.maximum(
                previous["max_hermiticity_defect"], max_hermiticity
            )
            max_negative_mass = real_xp.maximum(
                previous["max_negative_diagonal_mass"], max_negative_mass
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
        }
        self._last_rho_diagnostics[site] = diagnostics

        finite = xp.all(xp.isfinite(rhos), axis=(-2, -1)) & xp.isfinite(trace)
        valid_trace = trace > tolerance
        valid_negative = negative_max <= tolerance
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
        diagonal = real_xp.maximum(diagonal, real_xp.zeros_like(diagonal))
        diagnostics["clipped_negative_mass"] = negative_mass[-1]
        probabilities = diagonal / real_xp.sum(diagonal, axis=-1, keepdims=True)
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
            optimize=self.contraction_opt,
        )

    def _estimate_row_cache_bytes(self):
        """Conservatively bound dense row outputs before any cache is built."""
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
            for x in range(self.Lx):
                left = interfaces[x - 1] if x else 1
                right = interfaces[x] if x < self.Lx - 1 else 1
                physical = self._ket.ind_size(self._site_inds[x, y])
                # Local transfer, traced transfer, and suffix/prefix storage.
                row_elements += (physical**2 + 1) * left * right + left + right
            # Allow extra simultaneous intermediates and contraction workspace.
            largest_row = max(largest_row, 4 * row_elements * self._array_itemsize)
        self._row_cache_estimate_per_group = largest_row
        return largest_row

    def _use_row_cache(self, samples):
        """Select dense transfers only when their estimated work/storage fits."""
        estimated = (
            self._estimate_row_cache_bytes() * samples
            if self.row_cache_max_bytes else None
        )
        group_limit = 32 if self.Lx * self.Ly <= 9 else 4
        if self.row_cache_max_bytes == 0:
            reason = "disabled"
        elif estimated > self.row_cache_max_bytes:
            reason = "memory-budget"
        elif samples > group_limit:
            reason = "prefix-count"
        elif self.marginal_chi not in (None, 0) and self.Lx * self.Ly > 9:
            reason = "large-future"
        else:
            reason = "within-budget"
        self._last_cache_decision = {
            "estimated_cache_bytes": estimated,
            "cache_budget_bytes": self.row_cache_max_bytes,
            "cache_decision": reason,
        }
        return reason == "within-budget"

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
            # Trace the already-contracted local tensor instead of
            # contracting the whole column a second time.
            trace.append(local[-1].trace(ket_ind, bra_ind, preserve_tensor=True))

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
        rho = rho.data
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
        return qtn.tensor_contract(
            left, local,
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
            phi = self._update_conditioned_boundary(y, row_config, phi)

        self._last_boundary_mps = None if phi is None else phi.copy()
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": cache_builds,
            "site_prefix_updates": len(self.site_order),
            "mode": "transfer",
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

        amplitude = self._projected_amplitude(config)
        return config, self._log10_to_scaled(log10_proposal), self._scaled(amplitude)

    def _sample_one(self, rng):
        """Draw one configuration from the selected proposal backend."""
        if self.boundary_engine == "exact":
            return self._sample_one_exact(rng)

        config, omega = self._boundary_sample_or_probability(rng=rng)
        amplitude = self._projected_amplitude(config)
        return config, omega, self._scaled(amplitude)

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
                for value in np.unique(values):
                    shot_indices = group["indices"][values == value]
                    working = group["working"].copy()
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

        return groups, max_groups

    def _sample_batch_boundary(self, rng, samples):
        """Sample boundary proposals with one row cache per prefix group."""
        # Once almost every shot has a different post-row prefix, constructing
        # a dense transfer cache per group costs more than the original local
        # center contractions. Keep large production batches on that stable
        # path; small batches still exercise and benefit from row-cache reuse.
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
                        prefix_updates += 1
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
                    for value in np.unique(values):
                        shot_indices = group["indices"][values == value]
                        center = group["center"].copy()
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
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or int(samples) < 1
        ):
            raise ValueError("samples must be a positive integer.")
        samples = int(samples)
        self._reset_rho_diagnostics()
        self._grouped_draw_count = 0
        rng = self._make_rng(seed)

        if self.boundary_engine == "exact":
            groups, max_groups = self._sample_batch_exact(rng, samples)
        else:
            groups, max_groups = self._sample_batch_boundary(rng, samples)

        configs = [None] * samples
        omegas = [None] * samples
        amplitudes = [None] * samples
        for group in groups:
            omega = self._log10_to_scaled(group["log10"])
            amplitude = self._scaled(self._projected_amplitude(group["config"]))
            for index in group["indices"]:
                index = int(index)
                configs[index] = group["config"]
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
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or int(samples) < 1
        ):
            raise ValueError("samples must be a positive integer.")
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


class PepsBpSampler:
    """Sample PEPS configurations and amplitudes with BP proposals.

    The sampler draws configurations from :func:`sample_d2bp`, records the BP
    proposal probability ``omega(x)``, contracts the projected PEPS amplitude
    ``p(x)``, and returns the ingredients for the PEPS norm estimator
    ``E_q[|p(x)|^2 / omega(x)]``.

    Quimb's public D2BP sampler currently samples binary output indices. For a
    four-state spinful PEPS this class transparently samples two binary
    occupation legs per site and maps them back to the requested local
    fermion encoding. Symmray inputs are densified only in the private BP
    proposal copy; the original network remains block-sparse.
    """

    def __init__(
        self,
        tn,
        *,
        optimizer=None,
        sample_kwargs: dict[str, Any] | None = None,
        encoding=None,
        site_order=None,
    ):
        self.tn = getattr(tn, "tn", tn)
        self.Lx = self.tn.Lx
        self.Ly = self.tn.Ly
        self.optimizer = optimizer
        self.sample_kwargs = dict(sample_kwargs or {})
        self.encoding = encoding
        (
            self._bp_tn,
            self._split_inds,
            self.site_order,
            self._code_order,
        ) = _prepare_bp_binary_network(
            self.tn,
            site_order=site_order,
            encoding=encoding,
        )

    @staticmethod
    def mantissa_exponent10(w: float) -> tuple[float, int]:
        """Represent ``w`` as ``mantissa * 10**exponent``."""
        if w == 0:
            return 0.0, 0
        exponent = math.floor(math.log10(abs(w)))
        mantissa = w / (10 ** exponent)
        return mantissa, exponent

    def _get_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        global build_optimizer  # pylint: disable=global-statement
        if build_optimizer is None:
            from ..tensors.core import build_optimizer as _build_optimizer  # pylint: disable=import-outside-toplevel

            build_optimizer = _build_optimizer

        self.optimizer = build_optimizer(
            progbar=False,
            directory="cash",
            parallel=False,
            max_time="rate:1e8",
        )
        return self.optimizer

    def _config_list(self, config: dict[str, Any]) -> list[int]:
        if self._split_inds:
            out = []
            code_order = self._code_order
            # Invert the (up, down) -> physical-code map built above.
            code_from_bits = {
                (0, 0): code_order[0],
                (0, 1): code_order[1],
                (1, 0): code_order[2],
                (1, 1): code_order[3],
            }
            for site in self.site_order:
                up_ind, down_ind = self._split_inds[site]
                bits = (int(config[up_ind]), int(config[down_ind]))
                out.append(code_from_bits[bits])
            return out

        if self.site_order:
            out = []
            for site in self.site_order:
                try:
                    site_ind = self.tn.site_ind(site)
                except (AttributeError, KeyError):
                    site_ind = f"k{site[0]},{site[1]}"
                out.append(int(config[site_ind]))
            return out

        # Keep the small dummy-network/testing protocol backwards compatible.
        out = [None] * (self.Lx * self.Ly)
        for i in range(self.Lx):
            for j in range(self.Ly):
                out[i * self.Ly + j] = int(config[f"k{i},{j}"])
        return out

    def _sample_d2bp(self, sample_seed, bp_kwargs=None):
        kwargs = {
            "max_iterations": 100,
            "tol": 1.0e-2,
            "seed": sample_seed,
            "optimize": "auto-hq",
            "damping": 0.0,
            "diis": False,
            "update": "parallel",
            "local_convergence": True,
            "progbar": False,
        }
        kwargs.update(self.sample_kwargs)
        if bp_kwargs:
            kwargs.update(bp_kwargs)
        kwargs["seed"] = sample_seed

        global sample_d2bp  # pylint: disable=global-statement
        if sample_d2bp is None:
            from quimb.tensor.belief_propagation import sample_d2bp as _sample_d2bp  # pylint: disable=import-outside-toplevel

            sample_d2bp = _sample_d2bp

        return sample_d2bp(self._bp_tn, **kwargs)

    def _contract_sample(
        self,
        tn_flat,
        *,
        chi: int,
        method: str,
        max_separation: int,
        equalize_norms: bool,
        cutoff: float,
    ):
        optimizer = self._get_optimizer()

        def scaled_is_finite(value):
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return bool(np.isfinite(value[0]) and np.isfinite(value[1]))
            return bool(np.isfinite(value))

        def as_scaled(value):
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return value
            return self.mantissa_exponent10(value)

        if method == "mps":
            opts = {
                "optimize": optimizer,
                "strip_exponent": True,
            }
            result = call_quimb_2d(
                tn_flat.contract_boundary,
                max_bond=int(chi),
                mode="mps",
                final_contract_opts=opts,
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=["xmin", "xmax", "ymin", "ymax"],
                equalize_norms=equalize_norms,
                progbar=False,
            )
            if not scaled_is_finite(result):
                opts["strip_exponent"] = False
                result = call_quimb_2d(
                    tn_flat.contract_boundary,
                    max_bond=int(chi),
                    mode="mps",
                    final_contract_opts=opts,
                    max_separation=max_separation,
                    cutoff=cutoff,
                    sequence=["xmin", "xmax", "ymin", "ymax"],
                    equalize_norms=equalize_norms,
                    progbar=False,
                )
            return as_scaled(result)

        if method == "ctmrg":
            opts = {
                "optimize": optimizer,
                "strip_exponent": True,
            }
            result = call_quimb_2d(
                tn_flat.contract_ctmrg,
                max_bond=int(chi),
                final_contract_opts=opts,
                max_separation=max_separation,
                cutoff=cutoff,
                inplace=False,
                equalize_norms=equalize_norms,
                progbar=False,
            )
            if not scaled_is_finite(result):
                opts["strip_exponent"] = False
                result = call_quimb_2d(
                    tn_flat.contract_ctmrg,
                    max_bond=int(chi),
                    final_contract_opts=opts,
                    max_separation=max_separation,
                    cutoff=cutoff,
                    inplace=False,
                    equalize_norms=equalize_norms,
                    progbar=False,
                )
            return as_scaled(result)

        if method == "exact":
            result = tn_flat.contract(all, optimize=optimizer, strip_exponent=True)
            if not scaled_is_finite(result):
                result = tn_flat.contract(all, optimize=optimizer, strip_exponent=False)
            return as_scaled(result)

        raise ValueError(f"Unknown contraction method: {method!r}")

    def sample(
        self,
        *,
        chi: int = 12,
        samples: int = 1,
        method: str = "exact",
        seed: int | None = None,
        max_separation: int = 1,
        equalize_norms: bool = True,
        progbar: bool = False,
        cutoff: float = 0.0,
        bp_kwargs: dict[str, Any] | None = None,
    ) -> PEPSSampleResult:
        """Draw samples and contract the corresponding PEPS amplitudes.

        Parameters
        ----------
        bp_kwargs : dict, optional
            Override BP sampling parameters. Supported keys:
            max_iterations, tol, optimize, damping, diis, update,
            local_convergence, progbar.
        """
        configs: list[list[int]] = []
        omegas_mantissa: list[float] = []
        omegas_exponent: list[int] = []
        ps_mantissa: list[Any] = []
        ps_exponent: list[Any] = []

        sample_range: Iterable[int] = tqdm(
            range(int(samples)),
            desc="Sampling configs",
            disable=not progbar,
        )

        for sample_idx in sample_range:
            sample_seed = None if seed is None else int(seed) + sample_idx
            config_i, tn_flat, omega_i = self._sample_d2bp(sample_seed, bp_kwargs)
            mantissa, exponent = self._contract_sample(
                tn_flat,
                chi=chi,
                method=method,
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                cutoff=cutoff,
            )

            configs.append(self._config_list(config_i))

            omega_mantissa, omega_exponent = self.mantissa_exponent10(float(omega_i))
            omegas_mantissa.append(omega_mantissa)
            omegas_exponent.append(omega_exponent)

            ps_mantissa.append(mantissa)
            ps_exponent.append(exponent)

        return PEPSSampleResult(
            configs=configs,
            omegas=(omegas_mantissa, omegas_exponent),
            ps=(ps_mantissa, ps_exponent),
        )
