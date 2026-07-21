"""Samplers for MPS and PEPS tensor networks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

sample_d2bp = None
build_optimizer = None

__all__ = [
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
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown MpsSampler backend {backend!r}. Expected one of: {allowed}."
        ) from exc


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
    backend = _mps_array_backend(array)
    if backend == "torch":
        return array.detach().cpu().numpy()
    if backend == "cupy":
        return array.get()
    return np.asarray(array)


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
    detach = getattr(array, "detach", None)
    if callable(detach):
        array = detach()
        cpu = getattr(array, "cpu", None)
        if callable(cpu):
            array = cpu()
        numpy = getattr(array, "numpy", None)
        if callable(numpy):
            array = numpy()
    return np.asarray(array)


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


@dataclass
class MpsBatchSampleResult:
    """Backend-native batched MPS samples.

    Attributes
    ----------
    configs
        Array-like object with shape ``(n_samples, L)``. With the native
        sampler this can be a NumPy array, Torch tensor, or CuPy array.
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
        )

    def configs_1d(self) -> list[list[int]]:
        """Return configurations as Python ``list[list[int]]``."""
        configs = _backend_array_to_numpy(self.configs)
        return [[int(value) for value in config] for config in configs]

    def configs_2d(self) -> list[np.ndarray]:
        """Return configurations as ``(Ly, Lx)`` NumPy grids."""
        return self.to_sample_result().configs_2d

    def magnetizations(self, *, to_numpy: bool = False):
        """Per-sample magnetization ``(1 / L) * sum_i (1 - 2 * spin_i)``."""
        backend = _mps_array_backend(self.configs)
        if backend == "torch":
            configs = self.configs.to(dtype=self.probs.dtype)
            out = (1 - 2 * configs).sum(dim=1) / float(self.L)
            return out.detach().cpu().numpy() if to_numpy else out
        if backend == "cupy":
            configs = self.configs.astype(np.float64, copy=False)
            out = (1 - 2 * configs).sum(axis=1) / float(self.L)
            return out.get() if to_numpy else out
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
    backend : {"quimb", "native", "auto", "numpy", "torch", "cupy"}
        Sampling implementation. ``"quimb"`` preserves the historical CPU
        behavior. ``"native"`` accepts dense NumPy/Torch/CuPy tensors.
        ``"auto"`` tries native sampling and falls back to ``"quimb"`` when
        the MPS layout is unsupported.
    torch_compile : bool, default=False
        Opt into ``torch.compile`` for repeated, device-resident, unseeded
        Torch inference batches. Unsupported compiler environments and calls
        that need eager-only behavior fall back to eager sampling.

    Notes
    -----
    Native right environments are cached. Call :meth:`refresh` after changing
    the source MPS; otherwise the sampler continues to represent its previous
    tensor data.
    """

    def __init__(
        self,
        psi,
        one_d_to_two_d: dict[int, tuple[int, int]] | None = None,
        *,
        backend: str | None = "quimb",
        torch_compile: bool = False,
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
        self.resolved_backend = None
        self._source_psi = None
        self._native_arrays = None
        self._native_site_ops = None
        self._native_inference_site_ops = None
        self._evaluation_backend = None
        self._evaluation_arrays = None
        self._evaluation_site_ops = None
        self._psi = None
        self._torch_compiled_sample_fns = {}
        self._torch_compile_disabled = False

        self.refresh(psi)

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
        self._psi = None
        self._torch_compiled_sample_fns.clear()
        self._torch_compile_disabled = False

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
            lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
        )
        return self

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
            configs = configs.detach().cpu().numpy()
            probs_total = probs_total.detach().cpu().numpy()
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
        if to_numpy and backend == "cupy":
            configs = configs.get()
            probs_total = probs_total.get()
        if to_numpy:
            configs = np.asarray(configs)
            probs_total = np.asarray(probs_total)
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
        Set ``to_numpy=False`` to keep Torch/CuPy outputs on their device.
        """
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
        with user-supplied physical indices. It avoids looping over
        configurations and runs on Torch/CuPy when the MPS tensors do.
        """
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

        With ``backend="native"``, this returns backend-native arrays by
        default: Torch tensors stay on Torch and CuPy arrays stay on CuPy.
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
    ) -> MpsBatchSampleResult:
        """Draw samples and return a named batched result.

        This is the preferred API for fast downstream workflows. With
        ``backend="native"`` and ``to_numpy=False``, Torch/CuPy arrays stay on
        their current device. Use :meth:`sample_arrays` when tuple unpacking is
        more convenient, or :meth:`sample` when the legacy Python-list/grid
        result is needed.
        """
        configs, probs = self.sample_arrays(
            n_samples,
            seed=seed,
            to_numpy=to_numpy,
            track_grad=track_grad,
        )
        backend = (
            "numpy"
            if to_numpy or self._native_arrays is None
            else self.resolved_backend
        )
        return MpsBatchSampleResult(
            configs=configs,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
            one_d_to_two_d=dict(self.one_d_to_two_d),
            backend=backend,
        )

    def sample(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        *,
        track_grad: bool = False,
    ) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the MPS.

        The native backend uses batched conditional contractions on the MPS
        tensor device. The quimb backend uses ``MatrixProductState.sample()``,
        which internally right-canonicalizes the MPS and sweeps left-to-right.

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

        # Move to CPU if needed (cupy)
        if hasattr(vec, "get"):
            vec = vec.get()
        vec = np.asarray(vec, dtype=complex).ravel()
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
