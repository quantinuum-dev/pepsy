"""Samplers for MPS and PEPS tensor networks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

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
    "PepsBpSampler",
    "VecSampler",
]


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


def _configs_to_sample_result(configs, probs, *, Lx, Ly, one_d_to_two_d):
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
    )


@dataclass
class PEPSSampleResult:
    """Container for PEPS BP importance samples.

    Attributes
    ----------
    configs
        Sampled physical configurations in row-major ``[x * Ly + y]`` order.
    omegas
        Pair ``(mantissas, exponents)`` for BP proposal probabilities.
    ps
        Pair ``(mantissas, exponents)`` for sampled PEPS amplitudes.
    """

    configs: list[list[int]]
    omegas: tuple[list[float], list[int]]
    ps: tuple[list[Any], list[Any]]

    def __len__(self):
        """Return the number of sampled configurations."""
        return len(self.configs)


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
    """

    configs_1d: list[list[int]]
    configs_2d: list[np.ndarray]
    probs: list[float]
    Lx: int
    Ly: int

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
    """

    configs: Any
    probs: Any
    Lx: int
    Ly: int
    one_d_to_two_d: dict[int, tuple[int, int]]
    backend: str = "numpy"
    configuration_encoding: FermionConfigurationEncoding | None = None

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
            except Exception:
                if self.backend != "auto":
                    raise

        self.resolved_backend = "quimb"
        # Convert to numpy for quimb sampling compatibility
        self._psi = psi.copy()
        self._psi.apply_to_arrays(
            lambda x: ar.to_numpy(x)
        )
        return self

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

    def amplitudes(self, configs, *, to_numpy: bool = True):
        """Return batched MPS amplitudes for ``configs``.

        ``configs`` should have shape ``(batch, L)``. Dense NumPy, Torch, and
        CuPy MPS tensors are contracted in one batched backend-native pass.
        Symmray MPSs use block-sparse contractions on the underlying NumPy,
        Torch, or CuPy backend. Set ``to_numpy=False`` to keep Torch/CuPy
        outputs on their device.
        """
        if self._symmray_state is not None:
            out = self._symmray_amplitudes(configs)
            backend = self._symmray_state["array_backend"]
            return self._to_numpy_backend_array(out, backend) if to_numpy else out

        backend, site_data = self._get_evaluation_site_ops()
        if backend == "torch":
            out = self._torch_amplitudes(site_data, configs, L=self._L)
        else:
            out = self._array_namespace_amplitudes(
                site_data,
                configs,
                backend=backend,
                L=self._L,
            )
        return self._to_numpy_backend_array(out, backend) if to_numpy else out

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
    from the resulting categorical distribution.

    Parameters
    ----------
    state : TensorNetwork, Tensor, or array-like
        The dense state. If a quimb TensorNetwork/Tensor, the physical indices
        are assumed to follow ``ind_id`` format (default ``'k{}'``). If a raw
        array, it is reshaped to a 1D vector of length 2^L.
    one_d_to_two_d : dict[int, tuple[int, int]]
        Mapping from 1D site index to (x, y) lattice coordinate.
    ind_id : str
        Format string for physical index names (default ``'k{}'``).
    """

    def __init__(
        self,
        state,
        one_d_to_two_d: dict[int, tuple[int, int]],
        ind_id: str = "k{}",
    ):
        L = _validate_one_d_to_two_d(one_d_to_two_d)
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1

        # Extract the state vector with correct index ordering
        vec = self._to_vector(state, L, ind_id)

        vec = np.asarray(ar.to_numpy(vec), dtype=complex).ravel()
        expected_size = 2 ** L
        if vec.size != expected_size:
            raise ValueError(
                f"state vector size must be 2**L={expected_size} for L={L}; "
                f"got {vec.size}."
            )

        # Compute and cache Born probabilities
        probs = np.abs(vec) ** 2
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("state vector must have a finite non-zero norm.")
        probs /= total
        self._probs = probs
        self._L = L

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
        return np.asarray(state).reshape(-1)

    def sample(self, n_samples: int = 1, seed: int | None = None) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the state vector.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._probs), size=n_samples, p=self._probs)

        configs_1d = []
        configs_2d = []
        probs = []

        for idx in indices:
            # Convert flat index to binary configuration (big-endian: site 0 is MSB)
            config = [(idx >> (self._L - 1 - site)) & 1 for site in range(self._L)]
            configs_1d.append(config)

            grid = np.zeros((self.Ly, self.Lx), dtype=int)
            for site_1d, spin in enumerate(config):
                x, y = self.one_d_to_two_d[site_1d]
                grid[y, x] = spin
            configs_2d.append(grid)
            probs.append(float(self._probs[idx]))

        return MpsSampleResult(
            configs_1d=configs_1d,
            configs_2d=configs_2d,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
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
            result = tn_flat.contract_boundary(
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
                result = tn_flat.contract_boundary(
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
            result = tn_flat.contract_ctmrg(
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
                result = tn_flat.contract_ctmrg(
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
