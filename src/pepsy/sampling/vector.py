"""Exact dense-vector sampling in computational and Pauli bases."""

from __future__ import annotations

from numbers import Integral
from typing import Any
import numpy as np
from .results import (
    MpsBatchSampleResult,
    MpsSampleResult,
)
from ._common import (
    _backend_array_to_numpy,
    _mps_array_backend,
    _validate_one_d_to_two_d,
)

__all__ = ['VecSampler']


_TORCH_MULTINOMIAL_MAX_CATEGORIES = 1 << 24

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
        pepsy.sampling.results.MpsSampleResult
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
