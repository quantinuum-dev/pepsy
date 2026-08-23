"""Scalable sampling adapters for stabilizer tensor-network states."""

from __future__ import annotations

import numpy as np

from .samplers import (
    MpsBatchSampleResult,
    _basis_selection_probability,
    _configs_to_sample_result,
    _normalize_mps_sampler_backend,
    _resolve_measurement_basis,
    _validate_one_d_to_two_d,
    _validate_sample_chunk_size,
    _validate_sample_count,
)

__all__ = ["MpsStabSampler", "StabilizerMpsSampler"]


_DEFAULT_SAMPLE_CHUNK_SIZE = 4096


class MpsStabSampler:
    """Sample the physical STN state ``C|nu>`` without densifying it.

    The sampler accepts either a live :class:`MpsStabOptimizer` or the pair
    ``(C, nu)`` directly. It uses the tableau frame to map every requested
    local X/Y/Z measurement into a Pauli projector on the coefficient MPS.
    ``strategy="auto"`` currently resolves to this frame-projector path,
    which shares collapsed prefixes between shots just like the native MPS
    sampler. The optimizer state is never mutated by sampling.

    Parameters
    ----------
    state : MpsStabOptimizer or tableau simulator
        A live STN state representing ``C|nu>``, or the tableau ``C`` when
        ``nu`` is supplied as the second positional argument. Evolution can
        continue on a live optimizer and the sampler will observe the updated
        state.
    nu : MatrixProductState, optional
        Coefficient MPS ``|nu>`` when ``state`` is a tableau. Any additional
        keyword arguments are forwarded to
        :meth:`MpsStabOptimizer.from_tableau_and_state` in this form.
    one_d_to_two_d : dict[int, tuple[int, int]], optional
        Coordinate map used by the returned ``MpsSampleResult``-compatible
        helpers. By default logical qubit ``q`` maps to ``(q, 0)``.
    strategy : {"auto", "frame"}, default="auto"
        Sampling strategy. Both names select exact frame-mapped Pauli
        projectors; the explicit spelling makes the selected route visible to
        callers and leaves room for future dense/native alternatives.
    backend : {"auto", "native", "numpy", "torch", "cupy"}, default="auto"
        Backend for returned discrete arrays. ``"auto"`` and ``"native"``
        follow the coefficient-MPS backend when it is NumPy, Torch, or CuPy;
        unsupported structured backends fall back to NumPy. Explicit Torch or
        CuPy output requires the coefficient MPS to use that same backend.
    chunk_size : int, optional
        Sampling methods accept this per-call to bound temporary branch
        arrays. The final batch still contains all requested shots.
    """

    def __init__(
        self,
        state,
        nu=None,
        one_d_to_two_d=None,
        *,
        strategy="auto",
        backend="auto",
        **optimizer_kwargs,
    ):
        from ..optimizers.stabilizer_tn.mps_stab_optimizer import (
            MpsStabOptimizer,
        )

        # Preserve the original ``MpsStabSampler(optimizer, site_map)``
        # positional form while making ``MpsStabSampler(C, nu)`` natural.
        if isinstance(state, MpsStabOptimizer):
            if nu is not None:
                if one_d_to_two_d is None and isinstance(nu, dict):
                    one_d_to_two_d, nu = nu, None
                else:
                    raise TypeError(
                        "nu is only accepted when the first argument is a "
                        "tableau; pass one_d_to_two_d=... for an optimizer."
                    )
            if optimizer_kwargs:
                names = ", ".join(sorted(optimizer_kwargs))
                raise TypeError(
                    "Optimizer construction options cannot be supplied when "
                    f"passing a live MpsStabOptimizer: {names}."
                )
            optimizer = state
        else:
            if nu is None:
                raise TypeError(
                    "MpsStabSampler needs either an MpsStabOptimizer or "
                    "(tableau, nu) representing C|nu>."
                )
            optimizer = MpsStabOptimizer.from_tableau_and_state(
                state,
                nu,
                **optimizer_kwargs,
            )
        self._optimizer = optimizer
        self.strategy = self._normalize_strategy(strategy)
        self.resolved_strategy = "frame_pauli"
        self.backend = _normalize_mps_sampler_backend(backend)
        if self.backend in {"quimb", "symmray"}:
            raise ValueError(
                "MpsStabSampler supports backend='auto', 'native', 'numpy', "
                "'torch', or 'cupy'; quimb/symmray are not output backends."
            )
        self.resolved_backend = self._resolve_backend()
        self._set_site_map(one_d_to_two_d)

    def _resolve_backend(self):
        """Resolve the output backend against the live coefficient MPS."""
        info = self._optimizer.backend_info()
        live = str(info.get("array_backend", info.get("backend", "numpy")))
        if live not in {"numpy", "torch", "cupy"}:
            live = "numpy"
        if self.backend in {"auto", "native"}:
            return live
        if self.backend in {"torch", "cupy"} and live != self.backend:
            raise ValueError(
                f"MpsStabSampler backend={self.backend!r} requested, but the "
                f"coefficient MPS uses backend {live!r}; construct the optimizer "
                "with a matching to_backend converter or use backend='auto'."
            )
        return self.backend

    @property
    def native_backend(self):
        """Backend used for returned arrays after ``backend`` resolution."""
        return self.resolved_backend

    def _probability_dtype(self):
        dtype = str(getattr(self._optimizer, "backend_dtype", "complex128"))
        return np.float32 if "32" in dtype and "64" not in dtype else np.float64

    def _convert_arrays(self, configs, probs, *, to_numpy=False):
        """Convert a host-generated shot chunk to the selected native backend."""
        if to_numpy or self.resolved_backend == "numpy":
            return (
                np.asarray(configs, dtype=np.int8),
                np.asarray(probs, dtype=self._probability_dtype()),
            )
        if self.resolved_backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            device = getattr(self._optimizer, "backend_device", None) or "cpu"
            probability_dtype = (
                torch.float32
                if self._probability_dtype() == np.float32
                else torch.float64
            )
            return (
                torch.as_tensor(configs, dtype=torch.int8, device=device),
                torch.as_tensor(probs, dtype=probability_dtype, device=device),
            )
        if self.resolved_backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            return (
                cp.asarray(configs, dtype=cp.int8),
                cp.asarray(probs, dtype=cp.float32 if self._probability_dtype() == np.float32 else cp.float64),
            )
        raise RuntimeError(f"Unsupported resolved sampling backend {self.resolved_backend!r}.")

    @staticmethod
    def _configs_to_numpy(configs):
        """Move user-supplied native configurations to the host branch engine."""
        module = type(configs).__module__.split(".", 1)[0]
        if module == "torch":
            return configs.detach().cpu().numpy()
        if module == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            return cp.asnumpy(configs)
        return np.asarray(configs)

    def _allocate_arrays(self, n_samples, *, to_numpy):
        """Allocate final arrays once, matching ``MpsSampler`` chunk semantics."""
        backend = "numpy" if to_numpy else self.resolved_backend
        probability_dtype = self._probability_dtype()
        if backend == "numpy":
            return (
                np.empty((n_samples, self._L), dtype=np.int8),
                np.empty(n_samples, dtype=probability_dtype),
            )
        if backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            device = getattr(self._optimizer, "backend_device", None) or "cpu"
            return (
                torch.empty((n_samples, self._L), dtype=torch.int8, device=device),
                torch.empty(
                    n_samples,
                    dtype=torch.float32 if probability_dtype == np.float32 else torch.float64,
                    device=device,
                ),
            )
        if backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            dtype = cp.float32 if probability_dtype == np.float32 else cp.float64
            return cp.empty((n_samples, self._L), dtype=cp.int8), cp.empty(
                n_samples, dtype=dtype
            )
        raise RuntimeError(f"Unsupported sampling backend {backend!r}.")

    def _sample_arrays_resolved(
        self,
        n_samples,
        *,
        rng,
        resolved_basis,
        to_numpy,
        order,
        shuffle,
        chunk_size,
    ):
        configs, probs = self._allocate_arrays(n_samples, to_numpy=to_numpy)
        for start in range(0, n_samples, chunk_size):
            stop = min(n_samples, start + chunk_size)
            chunk_configs, chunk_probs, _ = self._optimizer._sample_basis_arrays(  # noqa: SLF001
                stop - start,
                basis=resolved_basis,
                seed=rng,
                order=order,
                shuffle=shuffle,
                resolved_basis=resolved_basis,
            )
            chunk_configs, chunk_probs = self._convert_arrays(
                chunk_configs,
                chunk_probs,
                to_numpy=to_numpy,
            )
            configs[start:stop] = chunk_configs
            probs[start:stop] = chunk_probs
        return configs, probs

    @staticmethod
    def _normalize_strategy(strategy):
        key = str(strategy).strip().lower().replace("-", "_")
        if key not in {"auto", "frame", "frame_pauli"}:
            raise ValueError(
                "Unknown MpsStabSampler strategy. Expected 'auto' or 'frame'."
            )
        return "auto" if key == "auto" else "frame"

    def _set_site_map(self, one_d_to_two_d):
        if one_d_to_two_d is None:
            one_d_to_two_d = {
                site: (site, 0) for site in range(self._optimizer.n)
            }
        self._L = _validate_one_d_to_two_d(
            one_d_to_two_d,
            expected_L=self._optimizer.n,
        )
        self.one_d_to_two_d = dict(one_d_to_two_d)
        self.Lx = max(x for x, _y in self.one_d_to_two_d.values()) + 1
        self.Ly = max(y for _x, y in self.one_d_to_two_d.values()) + 1

    @property
    def optimizer(self):
        """Return the live optimizer supplying the physical ``C|nu>`` state."""
        return self._optimizer

    @property
    def L(self):
        """Number of logical qubits represented by the sampled state."""
        return self._L

    def refresh(self, state=None):
        """Refresh the live optimizer reference while preserving the site map."""
        if state is None:
            return self
        from ..optimizers.stabilizer_tn.mps_stab_optimizer import (
            MpsStabOptimizer,
        )

        if not isinstance(state, MpsStabOptimizer):
            raise TypeError("state must be an MpsStabOptimizer.")
        if state.n != self._L:
            raise ValueError(
                f"Cannot refresh MpsStabSampler with n={state.n}; expected {self._L}."
            )
        self._optimizer = state
        self.resolved_backend = self._resolve_backend()
        return self

    @classmethod
    def from_tableau_and_state(
        cls,
        sim,
        nu,
        *,
        one_d_to_two_d=None,
        strategy="auto",
        backend="auto",
        **optimizer_kwargs,
    ):
        """Construct a sampler directly from a tableau ``C`` and coefficient ``nu``."""
        from ..optimizers.stabilizer_tn.mps_stab_optimizer import (
            MpsStabOptimizer,
        )

        optimizer = MpsStabOptimizer.from_tableau_and_state(
            sim,
            nu,
            **optimizer_kwargs,
        )
        return cls(
            optimizer,
            one_d_to_two_d=one_d_to_two_d,
            strategy=strategy,
            backend=backend,
        )

    def _basis_and_rng(self, basis, seed):
        rng = self._optimizer._sample_rng(seed)  # pylint: disable=protected-access
        resolved = _resolve_measurement_basis(basis, self._L, rng=rng)
        return resolved, rng

    def sample_arrays(
        self,
        n_samples=1,
        seed=None,
        *,
        to_numpy=False,
        track_grad=False,
        basis="Z",
        order=None,
        shuffle=True,
        chunk_size=_DEFAULT_SAMPLE_CHUNK_SIZE,
    ):
        """Draw backend-native configurations and exact Born probabilities.

        ``chunk_size`` bounds the temporary branch arrays used by each draw;
        the returned arrays still contain all ``n_samples`` rows. Set
        ``to_numpy=True`` to force CPU NumPy output even when the coefficient
        MPS is Torch- or CuPy-backed.
        """
        if track_grad:
            raise NotImplementedError(
                "MpsStabSampler frame-projector sampling does not retain gradients."
            )
        n_samples = _validate_sample_count(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        chunk_size = _validate_sample_chunk_size(chunk_size)
        if chunk_size is None:
            chunk_size = max(1, n_samples)
        resolved_basis, rng = self._basis_and_rng(basis, seed)
        return self._sample_arrays_resolved(
            n_samples,
            rng=rng,
            resolved_basis=resolved_basis,
            to_numpy=bool(to_numpy),
            order=order,
            shuffle=shuffle,
            chunk_size=chunk_size,
        )

    def sample_batch(
        self,
        n_samples=1,
        seed=None,
        *,
        to_numpy=False,
        track_grad=False,
        basis="Z",
        order=None,
        shuffle=True,
        chunk_size=_DEFAULT_SAMPLE_CHUNK_SIZE,
    ) -> MpsBatchSampleResult:
        """Draw a named backend-native batch, matching ``MpsSampler``."""
        if track_grad:
            raise NotImplementedError(
                "MpsStabSampler frame-projector sampling does not retain gradients."
            )
        n_samples = _validate_sample_count(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        chunk_size = _validate_sample_chunk_size(chunk_size)
        if chunk_size is None:
            chunk_size = max(1, n_samples)
        resolved_basis, rng = self._basis_and_rng(basis, seed)
        configs, probs = self._sample_arrays_resolved(
            n_samples,
            rng=rng,
            resolved_basis=resolved_basis,
            to_numpy=bool(to_numpy),
            order=order,
            shuffle=shuffle,
            chunk_size=chunk_size,
        )
        basis_probability = _basis_selection_probability(basis, self._L)
        weights = probs * basis_probability
        return MpsBatchSampleResult(
            configs=configs,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
            one_d_to_two_d=dict(self.one_d_to_two_d),
            backend="numpy" if to_numpy else self.resolved_backend,
            basis=resolved_basis,
            basis_probability=basis_probability,
            weights=weights,
        )

    def sample(
        self,
        n_samples=1,
        seed=None,
        *,
        track_grad=False,
        basis="Z",
        order=None,
        shuffle=True,
        chunk_size=_DEFAULT_SAMPLE_CHUNK_SIZE,
    ):
        """Draw configurations in the legacy ``MpsSampler`` result form."""
        batch = self.sample_batch(
            n_samples,
            seed=seed,
            track_grad=track_grad,
            basis=basis,
            order=order,
            shuffle=shuffle,
            chunk_size=chunk_size,
        )
        batch = batch.to_numpy()
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

    def probabilities(
        self, configs, *, to_numpy=True, basis="Z", order=None, seed=None
    ):
        """Return Born probabilities for supplied product-basis configs."""
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        resolved_basis, _rng = self._basis_and_rng(basis, seed)
        probabilities = self._optimizer.probability_bits_many(
            self._configs_to_numpy(configs),
            basis=resolved_basis,
            order=order,
            seed=seed,
        )
        if bool(to_numpy):
            return np.asarray(probabilities, dtype=self._probability_dtype())
        _configs, probabilities = self._convert_arrays(
            np.zeros((len(probabilities), self._L), dtype=np.int8),
            probabilities,
        )
        return probabilities

    def iter_samples(
        self,
        n_samples=1,
        seed=None,
        *,
        basis="Z",
        order=None,
        chunk_size=4096,
        shuffle=True,
    ):
        """Yield bounded-size ``MpsBatchSampleResult`` chunks."""
        n_samples = _validate_sample_count(n_samples)
        chunk_size = _validate_sample_chunk_size(chunk_size)
        if chunk_size is None:
            chunk_size = max(1, n_samples)
        resolved_basis, rng = self._basis_and_rng(basis, seed)
        for start in range(0, n_samples, chunk_size):
            count = min(chunk_size, n_samples - start)
            yield self.sample_batch(
                count,
                seed=rng,
                basis=resolved_basis,
                order=order,
                shuffle=shuffle,
                chunk_size=count,
            )


StabilizerMpsSampler = MpsStabSampler
