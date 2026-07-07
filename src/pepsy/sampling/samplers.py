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


class MpsSampler:
    """Sample from an MPS using quimb or a backend-native batched sampler.

    The legacy ``backend="quimb"`` path handles GPU→CPU conversion and calls
    quimb's canonical-form sampler. ``backend="native"`` keeps dense NumPy,
    Torch, or CuPy MPS arrays on their current device, canonicalizes once, and
    draws all requested samples with batched conditional contractions.

    Parameters
    ----------
    psi : MatrixProductState
        The MPS to sample from (can be on any backend).
    one_d_to_two_d : dict[int, tuple[int, int]]
        Mapping from 1D site index to (x, y) lattice coordinate.
    backend : {"quimb", "native", "auto", "numpy", "torch", "cupy"}
        Sampling implementation. ``"quimb"`` preserves the historical CPU
        behavior. ``"native"`` accepts dense NumPy/Torch/CuPy tensors.
        ``"auto"`` tries native sampling and falls back to ``"quimb"`` when
        the MPS layout is unsupported.
    """

    def __init__(
        self,
        psi,
        one_d_to_two_d: dict[int, tuple[int, int]],
        *,
        backend: str | None = "quimb",
    ):
        self._L = _validate_one_d_to_two_d(
            one_d_to_two_d,
            expected_L=getattr(psi, "L", None),
        )
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1
        self.backend = _normalize_mps_sampler_backend(backend)
        self.resolved_backend = None
        self._native_arrays = None
        self._evaluation_backend = None
        self._evaluation_arrays = None
        self._psi = None

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
                return
            except Exception:
                if self.backend != "auto":
                    raise

        self.resolved_backend = "quimb"
        # Convert to numpy for quimb sampling compatibility
        self._psi = psi.copy()
        self._psi.apply_to_arrays(
            lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
        )

    def _get_evaluation_arrays(self):
        if self._native_arrays is not None:
            return self.resolved_backend, self._native_arrays
        if self._evaluation_arrays is None:
            self._evaluation_backend, self._evaluation_arrays = (
                self._prepare_native_arrays(self._psi)
            )
        return self._evaluation_backend, self._evaluation_arrays

    @staticmethod
    def _canonicalize_for_native(psi):
        try:
            return psi.copy().canonicalize(0)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise ValueError(
                "backend-native MPS sampling requires a dense MPS that quimb can "
                "canonicalize on its current array backend."
            ) from exc

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
        psi_native = self._canonicalize_for_native(psi)
        arrays = tuple(
            self._site_array_lr_phys_r(psi_native, site)
            for site in range(psi_native.L)
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
    def _torch_sample(arrays, n_samples, seed):
        import torch  # pylint: disable=import-outside-toplevel

        device = arrays[0].device
        dtype = arrays[0].dtype
        if not (torch.is_floating_point(arrays[0]) or torch.is_complex(arrays[0])):
            dtype = torch.float64
        arrays = tuple(array.to(device=device, dtype=dtype) for array in arrays)
        vec = torch.ones((int(n_samples), 1), dtype=dtype, device=device)
        probs_total = torch.ones((int(n_samples),), dtype=torch.float64, device=device)
        configs = []
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        for array in arrays:
            amps = torch.einsum("bl,ldr->bdr", vec, array)
            probs = amps.abs().square().sum(dim=2).real
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(probs.dtype).tiny
            )
            choices = torch.multinomial(probs, 1, generator=generator).reshape(-1)
            batch = torch.arange(int(n_samples), device=device)
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / torch.sqrt(selected_probs).clamp_min(
                torch.finfo(selected_probs.dtype).tiny
            ).reshape(-1, 1).to(dtype=dtype)
            probs_total = probs_total * selected_probs.to(dtype=torch.float64)
            configs.append(choices)

        configs = torch.stack(configs, dim=1).detach().cpu().numpy()
        probs_total = probs_total.detach().cpu().numpy()
        return configs, probs_total

    @staticmethod
    def _array_namespace_sample(arrays, n_samples, seed, *, backend):
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        dtype = np.dtype(getattr(arrays[0], "dtype", np.float64))
        if dtype.kind not in {"f", "c"}:
            dtype = np.dtype(np.float64)
        arrays = tuple(array.astype(dtype, copy=False) for array in arrays)
        vec = xp.ones((int(n_samples), 1), dtype=dtype)
        probs_total = xp.ones((int(n_samples),), dtype=np.float64)
        configs = []
        rng = xp.random.default_rng(seed)

        for array in arrays:
            amps = xp.einsum("bl,ldr->bdr", vec, array)
            probs = xp.sum(xp.abs(amps) ** 2, axis=2).real
            probs = probs / xp.maximum(
                probs.sum(axis=1, keepdims=True),
                np.finfo(float).tiny,
            )
            cdf = xp.cumsum(probs, axis=1)
            draws = rng.random(int(n_samples))
            choices = xp.sum(draws[:, None] > cdf, axis=1).astype(np.int64)
            choices = xp.minimum(choices, probs.shape[1] - 1)
            batch = xp.arange(int(n_samples))
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / xp.sqrt(
                xp.maximum(selected_probs, np.finfo(float).tiny)
            )[:, None]
            probs_total = probs_total * selected_probs
            configs.append(choices)

        configs = xp.stack(configs, axis=1)
        if backend == "cupy":
            configs = configs.get()
            probs_total = probs_total.get()
        return np.asarray(configs), np.asarray(probs_total)

    def _native_sample_arrays(self, n_samples, seed):
        if self.resolved_backend == "torch":
            return self._torch_sample(self._native_arrays, n_samples, seed)
        return self._array_namespace_sample(
            self._native_arrays,
            n_samples,
            seed,
            backend=self.resolved_backend,
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
    def _torch_amplitudes(arrays, configs, *, L):
        import torch  # pylint: disable=import-outside-toplevel

        device = arrays[0].device
        dtype = arrays[0].dtype
        if not (torch.is_floating_point(arrays[0]) or torch.is_complex(arrays[0])):
            dtype = torch.float64
        arrays = tuple(array.to(device=device, dtype=dtype) for array in arrays)
        configs = MpsSampler._torch_configs(configs, device=device, L=L)
        vec = torch.ones((configs.shape[0], 1), dtype=dtype, device=device)
        batch = torch.arange(configs.shape[0], device=device)

        for site, array in enumerate(arrays):
            choices = configs[:, site]
            if bool(((choices < 0) | (choices >= array.shape[1])).any().item()):
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected = torch.index_select(array, 1, choices).permute(1, 0, 2)
            vec = torch.einsum("bl,blr->br", vec, selected)
            if vec.shape[0] != batch.shape[0]:  # pragma: no cover - sanity guard
                raise RuntimeError(
                    "Batched MPS amplitude contraction changed batch size."
                )
        return vec.reshape(-1)

    @staticmethod
    def _array_namespace_amplitudes(arrays, configs, *, backend, L):
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        dtype = np.dtype(getattr(arrays[0], "dtype", np.float64))
        if dtype.kind not in {"f", "c"}:
            dtype = np.dtype(np.float64)
        arrays = tuple(array.astype(dtype, copy=False) for array in arrays)
        configs = MpsSampler._array_namespace_configs(configs, backend=backend, L=L)
        vec = xp.ones((configs.shape[0], 1), dtype=dtype)

        for site, array in enumerate(arrays):
            choices = configs[:, site]
            invalid = (choices < 0) | (choices >= array.shape[1])
            invalid = (
                bool(invalid.any().get())
                if backend == "cupy"
                else bool(invalid.any())
            )
            if invalid:
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected = xp.moveaxis(array[:, choices, :], 1, 0)
            vec = xp.einsum("bl,blr->br", vec, selected)
        return vec.reshape(-1)

    @staticmethod
    def _torch_probabilities(arrays, configs, *, L):
        import torch  # pylint: disable=import-outside-toplevel

        device = arrays[0].device
        dtype = arrays[0].dtype
        if not (torch.is_floating_point(arrays[0]) or torch.is_complex(arrays[0])):
            dtype = torch.float64
        arrays = tuple(array.to(device=device, dtype=dtype) for array in arrays)
        configs = MpsSampler._torch_configs(configs, device=device, L=L)
        vec = torch.ones((configs.shape[0], 1), dtype=dtype, device=device)
        probs_total = torch.ones(
            (configs.shape[0],),
            dtype=torch.float64,
            device=device,
        )
        batch = torch.arange(configs.shape[0], device=device)

        for site, array in enumerate(arrays):
            amps = torch.einsum("bl,ldr->bdr", vec, array)
            probs = amps.abs().square().sum(dim=2).real
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(probs.dtype).tiny
            )
            choices = configs[:, site]
            if bool(((choices < 0) | (choices >= probs.shape[1])).any().item()):
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / torch.sqrt(selected_probs).clamp_min(
                torch.finfo(selected_probs.dtype).tiny
            ).reshape(-1, 1).to(dtype=dtype)
            probs_total = probs_total * selected_probs.to(dtype=torch.float64)
        return probs_total

    @staticmethod
    def _array_namespace_probabilities(arrays, configs, *, backend, L):
        xp = np
        if backend == "cupy":
            import cupy as xp  # pylint: disable=import-outside-toplevel,reimported

        dtype = np.dtype(getattr(arrays[0], "dtype", np.float64))
        if dtype.kind not in {"f", "c"}:
            dtype = np.dtype(np.float64)
        arrays = tuple(array.astype(dtype, copy=False) for array in arrays)
        configs = MpsSampler._array_namespace_configs(configs, backend=backend, L=L)
        vec = xp.ones((configs.shape[0], 1), dtype=dtype)
        probs_total = xp.ones((configs.shape[0],), dtype=np.float64)
        batch = xp.arange(configs.shape[0])

        for site, array in enumerate(arrays):
            amps = xp.einsum("bl,ldr->bdr", vec, array)
            probs = xp.sum(xp.abs(amps) ** 2, axis=2).real
            probs = probs / xp.maximum(
                probs.sum(axis=1, keepdims=True),
                np.finfo(float).tiny,
            )
            choices = configs[:, site]
            invalid = (choices < 0) | (choices >= probs.shape[1])
            invalid = (
                bool(invalid.any().get())
                if backend == "cupy"
                else bool(invalid.any())
            )
            if invalid:
                raise ValueError(
                    f"configs contain invalid physical index for site {site}."
                )
            selected_probs = probs[batch, choices]
            vec = amps[batch, choices, :] / xp.sqrt(
                xp.maximum(selected_probs, np.finfo(float).tiny)
            )[:, None]
            probs_total = probs_total * selected_probs
        return probs_total

    @staticmethod
    def _to_numpy_backend_array(array, backend):
        if backend == "torch":
            return array.detach().cpu().numpy()
        if backend == "cupy":
            return array.get()
        return np.asarray(array)

    def amplitudes(self, configs, *, to_numpy: bool = True):
        """Return batched MPS amplitudes for ``configs``.

        ``configs`` should have shape ``(batch, L)``. Dense NumPy, Torch, and
        CuPy MPS tensors are contracted in one batched backend-native pass.
        Set ``to_numpy=False`` to keep Torch/CuPy outputs on their device.
        """
        backend, arrays = self._get_evaluation_arrays()
        if backend == "torch":
            out = self._torch_amplitudes(arrays, configs, L=self._L)
        else:
            out = self._array_namespace_amplitudes(
                arrays,
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
        backend, arrays = self._get_evaluation_arrays()
        if backend == "torch":
            out = self._torch_probabilities(arrays, configs, L=self._L)
        else:
            out = self._array_namespace_probabilities(
                arrays,
                configs,
                backend=backend,
                L=self._L,
            )
        return self._to_numpy_backend_array(out, backend) if to_numpy else out

    def _result_from_arrays(self, configs_array, probs_array):
        configs_1d = []
        configs_2d = []
        probs = []
        for config, prob in zip(configs_array, probs_array):
            config = [int(value) for value in config]
            configs_1d.append(config)
            grid = np.zeros((self.Ly, self.Lx), dtype=int)
            for site_1d, spin in enumerate(config):
                x, y = self.one_d_to_two_d[site_1d]
                grid[y, x] = spin
            configs_2d.append(grid)
            probs.append(float(prob))
        return MpsSampleResult(
            configs_1d=configs_1d,
            configs_2d=configs_2d,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def sample(self, n_samples: int = 1, seed: int | None = None) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the MPS.

        The native backend uses batched conditional contractions on the MPS
        tensor device. The quimb backend uses ``MatrixProductState.sample()``,
        which internally right-canonicalizes the MPS and sweeps left-to-right.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples must be a positive integer.")
        if self._native_arrays is not None:
            configs, probs = self._native_sample_arrays(int(n_samples), seed)
            return self._result_from_arrays(configs, probs)

        configs_1d = []
        configs_2d = []
        probs = []

        for config, prob in self._psi.sample(n_samples, seed=seed):
            configs_1d.append(list(config))
            grid = np.zeros((self.Ly, self.Lx), dtype=int)
            for site_1d, spin in enumerate(config):
                x, y = self.one_d_to_two_d[site_1d]
                grid[y, x] = spin
            configs_2d.append(grid)
            probs.append(float(prob))

        return MpsSampleResult(
            configs_1d=configs_1d,
            configs_2d=configs_2d,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
        )


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
    """

    def __init__(self, tn, *, optimizer=None, sample_kwargs: dict[str, Any] | None = None):
        self.tn = tn
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.optimizer = optimizer
        self.sample_kwargs = dict(sample_kwargs or {})

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

        return sample_d2bp(self.tn, **kwargs)

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

        if method == "mps":
            return tn_flat.contract_boundary(
                max_bond=int(chi),
                mode="mps",
                final_contract_opts={"optimize": optimizer, "strip_exponent": True},
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=["xmin", "xmax", "ymin", "ymax"],
                equalize_norms=equalize_norms,
                progbar=False,
            )

        if method == "ctmrg":
            return tn_flat.contract_ctmrg(
                max_bond=int(chi),
                final_contract_opts={"optimize": optimizer, "strip_exponent": True},
                max_separation=max_separation,
                cutoff=cutoff,
                inplace=False,
                equalize_norms=equalize_norms,
                progbar=False,
            )

        if method == "exact":
            return tn_flat.contract(all, optimize=optimizer, strip_exponent=True)

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
