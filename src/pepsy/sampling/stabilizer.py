"""Scalable sampling adapters for stabilizer tensor-network states."""

from __future__ import annotations

from copy import deepcopy

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
_BRANCH_PROBABILITY_TOLERANCE = 1.0e-12


class MpsStabSampler:
    """Sample the physical STN state ``C|nu>`` without densifying it.

    The sampler accepts either a live :class:`MpsStabOptimizer` or the pair
    ``(C, nu)`` directly. It uses the tableau frame to map every requested
    local X/Y/Z measurement into a Pauli projector on the coefficient MPS.
    ``strategy="auto"`` currently resolves to this frame-projector path,
    which shares collapsed prefixes between shots just like the native MPS
    sampler. Set ``absorb_basis=True`` to use the basis-updating Lemma-3
    measurement on each copied branch. The optimizer state is never mutated by
    sampling.

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
        Sampling strategy. Both names currently select frame-mapped Pauli
        projectors; the explicit spelling makes the selected route visible to
        callers and leaves room for future alternatives.
    absorb_basis : bool, default=False
        If ``True``, use the basis-updating measurement path on each copied
        sampling branch. The frame is locally transformed so the measured
        Pauli becomes ``+/- Z_k``, then the coefficient site is projected.
        Localizing Clifford gates and any resulting projector compression use
        the ``chi``, ``mode``, and ``cutoff`` settings of the underlying
        optimizer. Since the frame changes after every absorbed measurement,
        frame images are recomputed per branch rather than cached globally.
    backend : {"auto", "native", "numpy", "torch", "cupy"}, default="auto"
        Backend for returned discrete arrays. ``"auto"`` and ``"native"``
        follow the coefficient-MPS backend when it is NumPy, Torch, or CuPy;
        unsupported structured backends fall back to NumPy. Explicit Torch or
        CuPy output requires the coefficient MPS to use that same backend.
    chunk_size : int, optional
        Sampling methods accept this per-call to bound temporary branch
        arrays. The final batch still contains all requested shots.
    **optimizer_kwargs
        Construction options forwarded to ``MpsStabOptimizer`` when ``state``
        is a tableau and ``nu`` is supplied. For example, use
        ``MpsStabSampler(C, nu, chi=16, mode="dmrg2", absorb_basis=True)``.
    """

    def __init__(
        self,
        state,
        nu=None,
        one_d_to_two_d=None,
        *,
        strategy="auto",
        absorb_basis=False,
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
        if not isinstance(absorb_basis, (bool, np.bool_)):
            raise TypeError("absorb_basis must be a boolean.")
        self.absorb_basis = bool(absorb_basis)
        self.resolved_strategy = (
            "frame_pauli_absorb" if self.absorb_basis else "frame_pauli"
        )
        self.backend = _normalize_mps_sampler_backend(backend)
        if self.backend in {"quimb", "symmray"}:
            raise ValueError(
                "MpsStabSampler supports backend='auto', 'native', 'numpy', "
                "'torch', or 'cupy'; quimb/symmray are not output backends."
            )
        self.resolved_backend = self._resolve_backend()
        self._set_site_map(one_d_to_two_d)
        # Sampling branches are temporary optimizer copies. Keep their
        # diagnostics on the sampler rather than changing the generic batch
        # result API or the live optimizer's state.
        self._last_sampling_diagnostics = []

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
        diagnostics,
    ):
        configs, probs = self._allocate_arrays(n_samples, to_numpy=to_numpy)
        for start in range(0, n_samples, chunk_size):
            stop = min(n_samples, start + chunk_size)
            chunk_configs, chunk_probs, _ = self._sample_basis_arrays(
                stop - start,
                basis=resolved_basis,
                seed=rng,
                order=order,
                shuffle=shuffle,
                resolved_basis=resolved_basis,
                diagnostics=diagnostics,
            )
            chunk_configs, chunk_probs = self._convert_arrays(
                chunk_configs,
                chunk_probs,
                to_numpy=to_numpy,
            )
            configs[start:stop] = chunk_configs
            probs[start:stop] = chunk_probs
        return configs, probs

    def _bit_measurement_order(self, order=None) -> tuple[int, ...]:
        """Return a validated physical-qubit order for computational readout."""
        optimizer = self._optimizer
        if order is None:
            order = "physical"
        if isinstance(order, str):
            key = order.strip().replace("-", "_").lower()
            if key in ("physical", "index", "default"):
                return tuple(range(self._L))
            if key in ("mps", "layout"):
                return tuple(int(q) for q in optimizer.logical_order)
            if key == "auto":
                return (
                    tuple(range(self._L))
                    if optimizer._layout_is_identity()  # noqa: SLF001
                    else tuple(int(q) for q in optimizer.logical_order)
                )
            raise ValueError(
                "order must be 'physical', 'mps', 'auto', or a permutation "
                f"of range({self._L}); got {order!r}."
            )
        try:
            order = tuple(int(q) for q in order)
        except TypeError as exc:
            raise TypeError(
                "order must be a string or a permutation of qubit indices."
            ) from exc
        if len(order) != self._L or sorted(order) != list(range(self._L)):
            raise ValueError(
                f"order must be a permutation of range({self._L}), got {order!r}."
            )
        return order

    @staticmethod
    def _prob_zero_from_expectation(exp: float) -> float:
        """Return clipped ``P(bit=0)`` for a computational-basis Z readout."""
        return min(max(0.5 * (1.0 + float(exp)), 0.0), 1.0)

    @staticmethod
    def _make_sampling_diagnostic(
        *,
        depth,
        qubit,
        axis,
        bit,
        prefix_bits,
        branch_size,
        prefix_probability,
        branch_probability,
        absorb_basis,
        condition_strategy,
        compression_events,
        norm_events,
    ):
        """Summarize one branch while keeping probability and loss separate."""
        compression_events = [dict(event) for event in compression_events]
        norm_events = [dict(event) for event in norm_events]

        # In absorb mode, these are the individual truncations incurred by
        # localizing the frame Pauli. Their product is the localizer's
        # retained norm, not a Born probability.
        localizer_fidelity = 1.0
        for event in compression_events:
            event_fidelity = event.get("local_fidelity")
            if event_fidelity is None:
                localizer_fidelity = None
                break
            localizer_fidelity *= float(event_fidelity)
        if localizer_fidelity is not None:
            localizer_fidelity = float(np.clip(localizer_fidelity, 0.0, 1.0))

        # Projector loss is recorded on the norm event. A deterministic frame
        # identity or exact one-site projection can legitimately have no
        # event, in which case its compression loss is zero.
        projector_event = next(
            (
                event
                for event in reversed(norm_events)
                if event.get("projector_survival") is not None
                or event.get("projector_infidelity") is not None
            ),
            None,
        )
        if projector_event is None:
            if any(not event.get("valid", True) for event in norm_events):
                # An invalid norm event means the state could not support a
                # reliable retained-norm estimate; report that explicitly.
                projector_fidelity = None
                projector_infidelity = None
            else:
                projector_fidelity = 1.0
                projector_infidelity = 0.0
        else:
            projector_fidelity = projector_event.get("projector_survival")
            projector_infidelity = projector_event.get("projector_infidelity")
            if projector_fidelity is None and projector_infidelity is not None:
                projector_fidelity = 1.0 - float(projector_infidelity)
            if projector_infidelity is None and projector_fidelity is not None:
                projector_infidelity = 1.0 - float(projector_fidelity)
            if projector_fidelity is not None:
                projector_fidelity = float(np.clip(projector_fidelity, 0.0, 1.0))
            if projector_infidelity is not None:
                projector_infidelity = float(np.clip(projector_infidelity, 0.0, 1.0))

        if localizer_fidelity is None or projector_fidelity is None:
            local_fidelity = None
            local_infidelity = None
            localizer_infidelity = None
        else:
            local_fidelity = float(
                np.clip(localizer_fidelity * projector_fidelity, 0.0, 1.0)
            )
            local_infidelity = 1.0 - local_fidelity
            localizer_infidelity = 1.0 - localizer_fidelity

        return {
            "depth": int(depth),
            "qubit": int(qubit),
            "basis": str(axis),
            "bit": int(bit),
            "outcome": 1 if int(bit) == 0 else -1,
            "prefix": tuple(int(value) for value in prefix_bits),
            "branch_size": int(branch_size),
            # Conditional Born probability of this selected bit.
            "branch_probability": float(branch_probability),
            # Probability of the complete prefix through this bit.
            "prefix_probability": float(prefix_probability),
            "joint_probability": float(prefix_probability * branch_probability),
            "absorb_basis": bool(absorb_basis),
            "condition_strategy": str(condition_strategy),
            "local_fidelity": local_fidelity,
            "local_infidelity": local_infidelity,
            "localizer_fidelity": localizer_fidelity,
            "localizer_infidelity": localizer_infidelity,
            "projector_fidelity": projector_fidelity,
            "projector_infidelity": projector_infidelity,
            # Keep every underlying event so callers can inspect each actual
            # truncation rather than only the aggregate conditional loss.
            "compression_events": compression_events,
            "norm_events": norm_events,
        }

    def get_sampling_diagnostics(self):
        """Return diagnostics for the most recently sampled batch.

        Each entry corresponds to one non-final conditional readout on one
        shared-prefix branch. ``branch_probability`` is a Born probability,
        while ``local_infidelity`` is retained-norm loss for that conditional
        update. The localizer and projector contributions are reported
        separately. The returned nested structure is an independent copy.

        For ``iter_samples`` and ``iter_sample_bits``, diagnostics refer to
        the most recently yielded chunk.
        """
        return deepcopy(self._last_sampling_diagnostics)

    @staticmethod
    def _bits_matrix(bitstrings, *, expected_length: int) -> np.ndarray:
        """Normalize one or more bitstrings to an ``(rows, n)`` int8 matrix."""
        from ..optimizers.stabilizer_tn.stn_state import _validate_bits

        if isinstance(bitstrings, str):
            rows = [_validate_bits(bitstrings, expected_length=expected_length)]
        else:
            arr = np.asarray(bitstrings)
            if arr.ndim == 2:
                rows = [
                    _validate_bits(row.tolist(), expected_length=expected_length)
                    for row in arr
                ]
            else:
                try:
                    values = list(bitstrings)
                except TypeError as exc:
                    raise TypeError(
                        "bitstrings must be a bitstring, a sequence of bitstrings, "
                        "or a 2D array-like of 0/1 values."
                    ) from exc
                if not values:
                    return np.empty((0, expected_length), dtype=np.int8)
                first = values[0]
                if isinstance(first, str):
                    rows = [
                        _validate_bits(row, expected_length=expected_length)
                        for row in values
                    ]
                else:
                    rows = [_validate_bits(values, expected_length=expected_length)]
        return np.asarray(rows, dtype=np.int8)

    @staticmethod
    def pack_bit_samples(samples) -> np.ndarray:
        """Pack an ``(shots, n)`` 0/1 sample array along the qubit axis."""
        arr = np.asarray(samples, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError("samples must be a 2D array of 0/1 bit values.")
        return np.packbits(arr, axis=1, bitorder="big")

    def _frame_terms(self, sim, axis, q):
        """Return the current coefficient-frame image for one physical readout."""
        return sim._frame_terms(axis, q)  # noqa: SLF001

    def _condition_probability_branch(
        self,
        sim,
        *,
        terms,
        sign,
        axis,
        q,
        bit,
        probability,
    ):
        """Condition a probability-query branch with the sampler fallback."""
        child = sim.copy()
        if not self.absorb_basis:
            child._condition_computational_bit(  # noqa: SLF001
                terms,
                sign,
                bit,
                probability=probability,
            )
            return child

        try:
            child._condition_absorbed_bit(  # noqa: SLF001
                axis,
                q,
                bit,
                probability=probability,
            )
        except ValueError as exc:
            if "~0-norm coefficient state" not in str(exc):
                raise
            # A finite-chi localizer can remove a branch that had nonzero
            # pre-localizer probability. Use the direct frame projector, just
            # as the shot sampler does, so both public query paths agree.
            child = sim.copy()
            child._condition_computational_bit(  # noqa: SLF001
                terms,
                sign,
                bit,
                probability=probability,
            )
        return child

    def _sample_basis_arrays(
        self,
        shots: int,
        *,
        basis="Z",
        seed=None,
        order=None,
        shuffle: bool = True,
        resolved_basis=None,
        diagnostics=None,
    ):
        """Sample product-Pauli outcomes using shared collapsed prefixes.

        The branching engine belongs to the sampler. The optimizer supplies
        only the state operations for evaluating and conditioning one branch.
        With ``absorb_basis=False`` the frame images are cached once because
        fixed-frame projectors leave ``C`` unchanged. With ``True``, each
        branch updates its own tableau, so the next frame image is recomputed
        from that branch's current ``C``.
        """
        shots = int(shots)
        if diagnostics is None:
            diagnostics = []
        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        rng = self._optimizer._sample_rng(seed)  # noqa: SLF001
        if resolved_basis is None:
            resolved_basis = _resolve_measurement_basis(basis, self._L, rng=rng)
        else:
            resolved_basis = tuple(str(axis).upper() for axis in resolved_basis)
            if len(resolved_basis) != self._L or any(
                axis not in {"X", "Y", "Z"} for axis in resolved_basis
            ):
                raise ValueError(
                    "resolved_basis must contain exactly one X/Y/Z axis per qubit."
                )
        order = self._bit_measurement_order(order)
        frame_terms = None
        if not self.absorb_basis:
            frame_terms = {
                int(q): self._frame_terms(self._optimizer, resolved_basis[int(q)], int(q))
                for q in order
            }
        bits = np.empty((shots, self._L), dtype=np.int8)
        probabilities = np.empty(shots, dtype=float)
        if shots == 0:
            return bits, probabilities, resolved_basis

        # Rows sharing a measured prefix share the collapsed coefficient MPS.
        stack = [(self._optimizer.copy(), 0, 0, shots, 1.0)]
        while stack:
            sim, pos, lo, hi, prefix_probability = stack.pop()
            q = order[pos]
            count = hi - lo
            if self.absorb_basis:
                terms, sign = self._frame_terms(sim, resolved_basis[q], q)
            else:
                terms, sign = frame_terms[q]
            expectation = sim._pauli_expectation(terms, sign)  # noqa: SLF001
            p0 = self._prob_zero_from_expectation(expectation)
            if p0 <= _BRANCH_PROBABILITY_TOLERANCE:
                n0 = 0
            elif p0 >= 1.0 - _BRANCH_PROBABILITY_TOLERANCE:
                n0 = count
            else:
                n0 = int(rng.binomial(count, p0))
            mid = lo + n0
            bits[lo:mid, q] = 0
            bits[mid:hi, q] = 1
            p_zero = prefix_probability * p0
            p_one = prefix_probability * (1.0 - p0)
            if pos + 1 == self._L:
                probabilities[lo:mid] = p_zero
                probabilities[mid:hi] = p_one
                continue
            live = ((0, lo, mid, p0), (1, mid, hi, 1.0 - p0))
            live = tuple(
                item
                for item in live
                if item[2] > item[1] and item[3] > _BRANCH_PROBABILITY_TOLERANCE
            )
            for branch_index, (bit, branch_lo, branch_hi, p_branch) in enumerate(live):
                # Absorption can be attempted on a private copy so a severe
                # finite-chi localizer approximation can be rolled back to
                # the mathematically equivalent direct frame projector.
                child = (
                    sim.copy()
                    if self.absorb_basis or branch_index < len(live) - 1
                    else sim
                )
                before_compression = len(child.get_compression_norm_events())
                before_norm = len(child.get_norm_events())
                condition_strategy = (
                    "absorbed" if self.absorb_basis else "frame_projector"
                )
                if self.absorb_basis:
                    try:
                        child._condition_absorbed_bit(  # noqa: SLF001
                            resolved_basis[q],
                            q,
                            bit,
                            probability=p_branch,
                        )
                    except ValueError as exc:
                        # A finite-chi localizer can rotate a nonzero sampled
                        # branch to a numerically zero Z branch. Re-run this
                        # child from the unmodified prefix with the direct
                        # C^dagger O C projector; this preserves a valid
                        # conditional state and leaves the fallback visible.
                        if "~0-norm coefficient state" not in str(exc):
                            raise
                        child = sim.copy()
                        child._condition_computational_bit(  # noqa: SLF001
                            terms,
                            sign,
                            bit,
                            probability=p_branch,
                        )
                        condition_strategy = "frame_projector_fallback"
                else:
                    child._condition_computational_bit(  # noqa: SLF001
                        terms,
                        sign,
                        bit,
                        probability=p_branch,
                    )
                compression_events = child.get_compression_norm_events()[
                    before_compression:
                ]
                norm_events = child.get_norm_events()[before_norm:]
                prefix_bits = tuple(
                    int(bits[branch_lo, previous_q])
                    for previous_q in order[:pos]
                ) + (int(bit),)
                diagnostics.append(
                    self._make_sampling_diagnostic(
                        depth=pos,
                        qubit=q,
                        axis=resolved_basis[q],
                        bit=bit,
                        prefix_bits=prefix_bits,
                        branch_size=branch_hi - branch_lo,
                        prefix_probability=prefix_probability,
                        branch_probability=p_branch,
                        absorb_basis=self.absorb_basis,
                        condition_strategy=condition_strategy,
                        compression_events=compression_events,
                        norm_events=norm_events,
                    )
                )
                stack.append(
                    (
                        child,
                        pos + 1,
                        branch_lo,
                        branch_hi,
                        prefix_probability * p_branch,
                    )
                )
        if shuffle:
            permutation = rng.permutation(shots)
            bits = bits[permutation]
            probabilities = probabilities[permutation]
        return bits, probabilities, resolved_basis

    def sample_bits(
        self,
        shots: int = 1,
        *,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
    ) -> np.ndarray:
        """Sample product-Pauli basis bitstrings without densifying ``C|nu>``.

        ``absorb_basis`` is selected when constructing this sampler. In that
        mode each collapsed branch updates its own tableau frame, while the
        underlying optimizer applies its configured finite-``chi``
        compression to every localizing Clifford or projector that needs it.
        """
        shots = _validate_sample_count(shots)
        if shots < 1:
            raise ValueError("shots must be a positive integer.")
        diagnostics = []
        out, _probs, _resolved_basis = self._sample_basis_arrays(
            shots,
            basis=basis,
            seed=seed,
            order=order,
            shuffle=shuffle,
            diagnostics=diagnostics,
        )
        self._last_sampling_diagnostics = diagnostics
        return self.pack_bit_samples(out) if packed else out

    def sample_basis(self, shots: int = 1, *, basis="Z", **kwargs):
        """Explicit alias for :meth:`sample_bits` with a Pauli-basis policy."""
        return self.sample_bits(shots, basis=basis, **kwargs)

    def sample_bitstrings(
        self,
        shots: int = 1,
        *,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
    ) -> np.ndarray:
        """Alias for :meth:`sample_bits` with an explicit bitstring name."""
        return self.sample_bits(
            shots,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
        )

    def probability_bits(self, bits, *, order=None, basis="Z", seed=None) -> float:
        """Return one product-Pauli outcome probability by chain-rule collapse.

        The same finite-``chi`` absorbed-basis fallback as sampling is used
        when a localizer produces a numerical zero branch.
        """
        from ..optimizers.stabilizer_tn.stn_state import _validate_bits

        bits = _validate_bits(bits, expected_length=self._L)
        rng = self._optimizer._sample_rng(seed)  # noqa: SLF001
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=rng)
        order = self._bit_measurement_order(order)
        frame_terms = None
        if not self.absorb_basis:
            frame_terms = {
                int(q): self._frame_terms(self._optimizer, resolved_basis[int(q)], int(q))
                for q in order
            }
        tmp = self._optimizer.copy()
        probability = 1.0
        for q in order:
            if self.absorb_basis:
                terms, sign = self._frame_terms(tmp, resolved_basis[q], q)
            else:
                terms, sign = frame_terms[q]
            expectation = tmp._pauli_expectation(terms, sign)  # noqa: SLF001
            p0 = self._prob_zero_from_expectation(expectation)
            bit = int(bits[q])
            branch_probability = p0 if bit == 0 else 1.0 - p0
            if branch_probability <= _BRANCH_PROBABILITY_TOLERANCE:
                return 0.0
            probability *= branch_probability
            tmp = self._condition_probability_branch(
                tmp,
                terms=terms,
                sign=sign,
                axis=resolved_basis[q],
                q=q,
                bit=bit,
                probability=branch_probability,
            )
        return float(probability)

    def probability_bits_many(
        self, bitstrings, *, order=None, basis="Z", seed=None
    ) -> np.ndarray:
        """Return probabilities for many product-Pauli basis bitstrings."""
        bits = self._bits_matrix(bitstrings, expected_length=self._L)
        probabilities = np.zeros(bits.shape[0], dtype=float)
        if bits.shape[0] == 0:
            return probabilities
        rng = self._optimizer._sample_rng(seed)  # noqa: SLF001
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=rng)
        order = self._bit_measurement_order(order)
        frame_terms = None
        if not self.absorb_basis:
            frame_terms = {
                int(q): self._frame_terms(self._optimizer, resolved_basis[int(q)], int(q))
                for q in order
            }
        stack = [(self._optimizer.copy(), 0, np.arange(bits.shape[0]), 1.0)]
        while stack:
            sim, pos, indices, prefix_probability = stack.pop()
            if indices.size == 0:
                continue
            q = order[pos]
            if self.absorb_basis:
                terms, sign = self._frame_terms(sim, resolved_basis[q], q)
            else:
                terms, sign = frame_terms[q]
            expectation = sim._pauli_expectation(terms, sign)  # noqa: SLF001
            p0 = self._prob_zero_from_expectation(expectation)
            branch_specs = (
                (0, indices[bits[indices, q] == 0], p0),
                (1, indices[bits[indices, q] == 1], 1.0 - p0),
            )
            live = tuple(
                (bit, idx, float(p_branch))
                for bit, idx, p_branch in branch_specs
                if idx.size > 0 and p_branch > _BRANCH_PROBABILITY_TOLERANCE
            )
            for _bit, idx, p_branch in branch_specs:
                if idx.size > 0 and p_branch <= _BRANCH_PROBABILITY_TOLERANCE:
                    probabilities[idx] = 0.0
            for bit, idx, p_branch in live:
                branch_probability = prefix_probability * p_branch
                if pos + 1 == self._L:
                    probabilities[idx] = branch_probability
                    continue
                child = self._condition_probability_branch(
                    sim,
                    terms=terms,
                    sign=sign,
                    axis=resolved_basis[q],
                    q=q,
                    bit=bit,
                    probability=p_branch,
                )
                stack.append(
                    (child, pos + 1, idx, branch_probability)
                )
        return probabilities

    def bitstring_probability(self, bits, *, order=None, basis="Z", seed=None) -> float:
        """Alias for :meth:`probability_bits`."""
        return self.probability_bits(bits, order=order, basis=basis, seed=seed)

    def bitstring_probabilities(
        self, bitstrings, *, order=None, basis="Z", seed=None
    ) -> np.ndarray:
        """Alias for :meth:`probability_bits_many`."""
        return self.probability_bits_many(
            bitstrings, order=order, basis=basis, seed=seed
        )

    def iter_sample_bits(
        self,
        shots: int,
        *,
        chunk_size: int,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
    ):
        """Yield raw sampled bit arrays in bounded-size chunks."""
        shots = int(shots)
        chunk_size = int(chunk_size)
        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        rng = self._optimizer._sample_rng(seed)  # noqa: SLF001
        resolved_basis = _resolve_measurement_basis(basis, self._L, rng=rng)
        done = 0
        while done < shots:
            take = min(chunk_size, shots - done)
            yield self.sample_bits(
                take,
                seed=rng,
                order=order,
                shuffle=shuffle,
                packed=packed,
                basis=resolved_basis,
            )
            done += take

    def iter_sample_bitstrings(
        self,
        shots: int,
        *,
        chunk_size: int,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
    ):
        """Alias for :meth:`iter_sample_bits`."""
        yield from self.iter_sample_bits(
            shots,
            chunk_size=chunk_size,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
        )

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

    @property
    def chi(self):
        """Bond cap used by the underlying coefficient-MPS optimizer."""
        return self._optimizer.chi

    @property
    def cutoff(self):
        """Singular-value cutoff used by the underlying optimizer."""
        return self._optimizer.cutoff

    @property
    def mode(self):
        """Compression mode used by the underlying optimizer."""
        return self._optimizer.mode

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
        absorb_basis=False,
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
            absorb_basis=absorb_basis,
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
        """Draw configurations and conditional Born probabilities.

        ``chunk_size`` bounds the temporary branch arrays used by each draw;
        the returned arrays still contain all ``n_samples`` rows. Set
        ``to_numpy=True`` to force CPU NumPy output even when the coefficient
        MPS is Torch- or CuPy-backed.

        With ``chi=None``, the probabilities are exact up to numerical
        precision. With finite ``chi``, conditional branches are compressed,
        so the probabilities describe the compressed sampler state; inspect
        :meth:`get_sampling_diagnostics` for the associated norm-loss signals.
        Sampling is inference-only and does not support ``track_grad=True``.
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
        diagnostics = []
        result = self._sample_arrays_resolved(
            n_samples,
            rng=rng,
            resolved_basis=resolved_basis,
            to_numpy=bool(to_numpy),
            order=order,
            shuffle=shuffle,
            chunk_size=chunk_size,
            diagnostics=diagnostics,
        )
        self._last_sampling_diagnostics = diagnostics
        return result

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
        """Draw a named backend-native batch, matching ``MpsSampler``.

        The ``probs`` field is ``p(config | resolved_basis)``. It is exact for
        an uncompressed coefficient MPS and reflects the compressed
        conditional branches when finite ``chi`` is configured.
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
        diagnostics = []
        configs, probs = self._sample_arrays_resolved(
            n_samples,
            rng=rng,
            resolved_basis=resolved_basis,
            to_numpy=bool(to_numpy),
            order=order,
            shuffle=shuffle,
            chunk_size=chunk_size,
            diagnostics=diagnostics,
        )
        self._last_sampling_diagnostics = diagnostics
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
        """Return conditional Born probabilities for product-basis configs."""
        if not isinstance(to_numpy, (bool, np.bool_)):
            raise TypeError("to_numpy must be a boolean.")
        resolved_basis, _rng = self._basis_and_rng(basis, seed)
        probabilities = self.probability_bits_many(
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
