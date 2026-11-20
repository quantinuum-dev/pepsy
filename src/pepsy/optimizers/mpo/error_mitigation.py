"""Tensor-network error mitigation from randomized Pauli readouts."""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import product

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..._internal.quimb import run_seeded_quimb
from ..._internal.validation import normalize_integer, normalize_integer_tuple
from ...backends import reg_stop_gradient_torch
from ...backends.config import _patch_backend_string_namespace_key
from ...backends.convert import infer_backend_converter_from_sample
from ...operators.mpo_product import _fit_lazy_mpo_target
from ...operators.mpo_semantic import _resolve_compression_cutoff, _resolve_compression_cutoff_mode
from ...sampling import MpsSampler, VecSampler
from ..mps import MpsOptimizer
from ..mps._streams import _resolve_symbolic_gate_entry
from ..noise import (
    CoalescedTrajectoryResult,
    TrajectoryChannel,
    TrajectoryEvent,
    _as_entries,
    _is_unitary_matrix,
    _trajectory_event_from_stochastic_entry,
    compile_trajectory_stream,
    run_trajectory_shots,
)
from .optimizer import MpoOptimizer

__all__ = ["MpoErrorMit", "PauliShots", "TemEstimate"]

_PAULIS = np.array([
    [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
])
_PTM_BASES = {1: _PAULIS, 2: np.array([np.kron(a, b) for a, b in product(_PAULIS, repeat=2)])}
_READOUT_ROTATIONS = np.array([
    [[1, 1], [1, -1]], [[1, -1j], [1, 1j]],
]) / np.sqrt(2)


def _positive_integer(value, name):
    value = normalize_integer(value, name=name)
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _readout_codes(values, alphabet, name):
    codes = np.asarray(values)
    if codes.ndim <= 1 and codes.dtype.kind in "US":
        rows = np.atleast_1d(codes.astype(str))
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError(f"{name} strings must have equal length.")
        codes = np.array([list(row) for row in rows])
    if codes.ndim != 2 or not all(codes.shape):
        raise ValueError(f"{name} must contain one nonempty row per readout.")
    if codes.dtype.kind in "US":
        symbols = np.char.upper(codes.astype(str))
        codes = np.full(symbols.shape, -1, dtype=np.int8)
        for index, symbol in enumerate(alphabet):
            codes[symbols == symbol] = index
    if codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= len(alphabet)):
        raise ValueError(f"{name} must use {alphabet} or their zero-based integer codes.")
    return codes.astype(np.int8, copy=True)


def _basis_probabilities(probs, n_qubits):
    probs = np.asarray(probs, dtype=float)
    if probs.shape == (3,):
        probs = np.broadcast_to(probs, (n_qubits, 3))
    if (
        probs.shape != (n_qubits, 3)
        or not np.all(np.isfinite(probs))
        or np.any(probs <= 0)
        or not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("basis_probs must give positive X, Y, Z probabilities summing to one per qubit.")
    return probs / probs.sum(axis=1, keepdims=True)


@dataclass(frozen=True, eq=False)
class PauliShots:
    """Independent product-basis measurements, optionally counted by readout.

    Supply bases as strings such as ``["XY", "ZX"]`` or integer codes
    X=0, Y=1, Z=2. Supply bits as strings or integer rows with 0 for the +1
    outcome. Columns are in logical qubit order. ``basis_probs`` specifies
    the sampling design, either one X/Y/Z distribution or one per qubit.
    ``counts`` groups identical readouts with positive multiplicities.
    """

    bases: object
    bits: object
    counts: object = None
    basis_probs: object = (1 / 3, 1 / 3, 1 / 3)

    def __post_init__(self):
        bases = _readout_codes(self.bases, "XYZ", "bases")
        bits = _readout_codes(self.bits, "01", "bits")
        if bases.shape != bits.shape:
            raise ValueError("bases and bits must have the same shape.")
        counts = np.ones(len(bases), dtype=np.int64) if self.counts is None else np.asarray(self.counts)
        if counts.shape != (len(bases),) or counts.dtype.kind not in "iu" or np.any(counts <= 0):
            raise ValueError("counts must give one positive integer per readout.")
        for name, values in (
            ("bases", bases), ("bits", bits), ("counts", counts.copy()),
            ("basis_probs", _basis_probabilities(self.basis_probs, bases.shape[1])),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def shots(self):
        """Number of measurements represented by the records."""
        return sum(map(int, self.counts))


@dataclass(frozen=True)
class TemEstimate:
    """Raw and mitigated means and their sampling standard errors."""

    raw: float
    mitigated: float
    raw_stderr: float
    mitigated_stderr: float


def _check_support(where):
    where = normalize_integer_tuple(where, name="where")
    if len(where) not in {1, 2} or len(where) != len(set(where)) or min(where) < 0:
        raise ValueError("TEM events must act on one or two distinct nonnegative qubit indices.")
    return where


def _matrix(value, where, *, require_unitary=False):
    dim = 2 ** len(where)
    source = np.asarray(ar.to_numpy(value))
    matrix = source.astype(complex, copy=True)
    if matrix.shape not in {(dim, dim), (2,) * (2 * len(where))} or not np.all(np.isfinite(matrix)):
        raise ValueError("TEM matrices must be finite qubit operators matching their support.")
    matrix = matrix.reshape(dim, dim)
    if require_unitary:
        # Promotion improves the Gram calculation but retains input roundoff.
        tolerance = 1e-10
        if source.dtype.type in (np.float32, np.complex64):
            tolerance = 8.0 * float(np.finfo(np.float32).eps)
        if not _is_unitary_matrix(matrix, atol=tolerance):
            raise ValueError("TEM unitary layers require unitary gates.")
    return matrix


def _transfer(operators, weights, where):
    basis = _PTM_BASES[len(where)]
    images = sum(
        weight * (op @ basis @ op.conj().T)
        for op, weight in zip(operators, weights, strict=True)
    )
    return np.einsum("aij,bji->ab", basis, images).real / basis.shape[1]


def _layer_events(gates, noise):
    for gate in _as_entries(gates):
        resolved = _resolve_symbolic_gate_entry(gate, None)
        if not isinstance(resolved, (tuple, list)) or len(resolved) != 2 or isinstance(resolved[0], str):
            raise ValueError("TEM unitary layers accept named gates or (matrix, where) entries.")
        matrix, where = resolved
        where = _check_support(where)
        matrix = _matrix(matrix, where, require_unitary=True)
        yield (matrix, where), _transfer((matrix,), (1,), where), where, False

    events = (noise,) if isinstance(noise, TrajectoryEvent) else _as_entries(noise)
    for event in events:
        if not isinstance(event, TrajectoryEvent):
            event = _trajectory_event_from_stochastic_entry(event)
        if event is None:
            raise ValueError("TEM noise layers accept named channels or TrajectoryEvent objects.")
        where = _check_support(event.where)
        channel = event.channel
        weights = [outcome.probability if channel.mode == "mixture" else 1 for outcome in channel.outcomes]
        operators = [_matrix(outcome.gate, where) for outcome in channel.outcomes]
        transfer = _transfer(operators, weights, where)
        if np.linalg.matrix_rank(transfer) < len(transfer):
            raise ValueError(f"The noise channel on {where} has a singular Pauli transfer matrix.")
        if channel.mode == "mixture":
            owned = TrajectoryChannel.mixture(
                (outcome.label, weight, operator)
                for outcome, weight, operator in zip(channel.outcomes, weights, operators, strict=True)
            )
        else:
            owned = TrajectoryChannel.kraus(
                (outcome.label, operator)
                for outcome, operator in zip(channel.outcomes, operators, strict=True)
            )
        yield TrajectoryEvent(owned, where), np.linalg.inv(transfer), where, True


def _stream_layers(gate_stream):
    plan = compile_trajectory_stream(gate_stream)
    if plan.has_controls or plan.has_leakage:
        raise ValueError("TEM requires a fixed qubit circuit of unitary gates and invertible channels.")
    gates, noise = [], []
    for event in plan.entries:
        if isinstance(event, TrajectoryEvent):
            noise.append(event)
        else:
            if noise:
                yield gates, noise
                gates, noise = [], []
            gates.append(event)
    if gates or noise:
        yield gates, noise


def _gate_transfer(target, matrix, where, index):
    inds = tuple(index(site) for site in where)
    # Factor only the local map, retaining its numerical Schmidt rank.
    qtn.tensor_network_gate_inds(
        target, matrix, inds, contract="split-gate" if len(where) == 2 else False,
        cutoff=np.finfo(ar.get_dtype_name(matrix)).eps * 16, cutoff_mode="rel", inplace=True,
    )
    for site, ind in zip(where, inds, strict=True):
        tid, = target.ind_map[ind]
        target.tensor_map[tid].add_tag(target.site_tag(site))


def _ptm_cores(mpo):
    for site in range(mpo.L):
        left = (mpo.bond(site - 1, site),) if site else ()
        right = (mpo.bond(site, site + 1),) if site + 1 < mpo.L else ()
        tensor = mpo[site].transpose(*left, *right, mpo.upper_ind(site), mpo.lower_ind(site))
        yield tensor.data.reshape(
            (tensor.ind_size(left[0]) if left else 1, tensor.ind_size(right[0]) if right else 1, 4, 4),
        )


def _shot_stats(scores, counts, total):
    mean = float(np.average(scores, weights=counts))
    if total == 1:
        return mean, float("nan")
    variance = np.dot(counts, np.square(scores - mean)) / (total * (total - 1.0))
    return mean, float(np.sqrt(variance))


def _trajectory_readouts(trajectories, thresholds, rng):
    if isinstance(trajectories, CoalescedTrajectoryResult):
        leaves = ((leaf.optimizer, leaf.count) for leaf in trajectories.leaves)
    else:
        leaves = ((optimizer, 1) for optimizer in trajectories.optimizers)
    for optimizer, count in leaves:
        uniforms = rng.random((count, len(thresholds), 1))
        bases = (uniforms > thresholds).sum(axis=-1)
        patterns, counts = np.unique(bases, axis=0, return_counts=True)
        vector_sampler = (
            VecSampler(optimizer.to_dense().reshape(-1))
            if optimizer.mode in {"exact", "exact-batch"} else None
        )
        if vector_sampler is None:
            to_state = infer_backend_converter_from_sample(optimizer.p.arrays[0])
            rotations = tuple(map(to_state, _READOUT_ROTATIONS))
        for pattern, multiplicity in zip(patterns, counts, strict=True):
            seed = int(rng.integers(2**32))
            if vector_sampler is None:
                rotated = optimizer.p.copy()
                for site, logical in enumerate(optimizer.logical_order):
                    axis = pattern[logical]
                    if axis < 2:
                        rotated.gate_(rotations[axis], site, contract=True)
                sample = MpsSampler(rotated, backend="auto").sample_batch(
                    int(multiplicity), seed=seed, to_numpy=True,
                )
                bits = optimizer.remap_sample(sample.configs)
            else:
                sample = vector_sampler.sample_batch(
                    int(multiplicity), seed=seed, to_numpy=True,
                    basis="".join("XYZ"[axis] for axis in pattern),
                )
                bits = sample.configs
            outcomes, outcome_counts = np.unique(bits, axis=0, return_counts=True)
            for bits, frequency in zip(outcomes, outcome_counts, strict=True):
                yield (*pattern, *bits), int(frequency)


class MpoErrorMit:
    """Build a middle-out mitigation MPO for a noisy qubit circuit.

    Pass an MPS gate stream or explicit ``layers=[(gates, noise), ...]``.
    In a stream, a run of consecutive noise events closes the current layer.
    Noise events must be invertible maps.
    ``chi=None`` keeps the full mitigation MPO, for small exact references.
    Local PTMs are prepared in NumPy. Supply ``to_backend`` to place the
    mitigation MPO and shot-score contractions on a dense Torch device.
    """

    def __init__(
        self, gate_stream=None, *, layers=None, n_qubits=None, chi=32, mode="dmrg2", cutoff=1e-12,
        cutoff_mode="auto", n_iter=2, fit_init_seed=0, to_backend=None,
    ):
        if gate_stream is not None and layers is not None:
            raise ValueError("Pass gate_stream or layers.")
        if layers is None:
            layers = _stream_layers(gate_stream)
        compiled = tuple(tuple(_layer_events(gates, noise)) for gates, noise in layers)
        needed = 1 + max((max(where) for layer in compiled for _, _, where, _ in layer), default=-1)
        self._n_qubits = _positive_integer(needed if n_qubits is None else n_qubits, "n_qubits")
        if self.n_qubits < needed:
            raise ValueError("n_qubits must cover all circuit events.")
        self._chi = None if chi is None else _positive_integer(chi, "chi")
        mode = str(mode).strip().lower()
        self._block_size = {"fit": 1, "dmrg": 1, "dmrg1": 1, "dmrg2": 2, "dmrg3": 3}.get(mode)
        if self._block_size is not None or mode == "svd":
            self._method = "direct"
        elif MpoOptimizer._is_mpo_mode(mode):
            self._method = MpoOptimizer._mode_mpo_method(mode)
        else:
            raise ValueError("mode must select a DMRG or Quimb MPO compressor.")
        self._cutoff = _resolve_compression_cutoff(cutoff, np.empty((), dtype=float))
        self._cutoff_mode = _resolve_compression_cutoff_mode(cutoff_mode)
        self._n_iter = _positive_integer(n_iter, "n_iter")
        self._fit_init_seed = normalize_integer(fit_init_seed, name="fit_init_seed")
        if self._fit_init_seed < 0:
            raise ValueError("fit_init_seed must be nonnegative.")
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        self._to_backend = np.asarray if to_backend is None else to_backend
        self._layers = tuple(
            tuple((matrix, where, is_noise) for _, matrix, where, is_noise in layer) for layer in compiled
        )
        self._stream = compile_trajectory_stream(
            tuple(event for layer in compiled for event, _, _, _ in layer)
        )

    @property
    def n_qubits(self):
        """Number of logical qubits in the compiled circuit."""
        return self._n_qubits

    @cached_property
    def _mitigation(self):
        identity = self._to_backend(np.eye(4))
        if ar.infer_backend(identity) == "torch":
            _patch_backend_string_namespace_key()
        mpo = qtn.MPO_product_operator([identity] * self.n_qubits)
        has_noise = False
        for step, layer in enumerate(self._layers):
            if not layer or (not has_noise and not any(is_noise for _, _, is_noise in layer)):
                continue
            target = mpo.copy()
            for matrix, where, is_noise in layer:
                if is_noise:
                    # A gate on the lower legs right-multiplies by its transpose.
                    _gate_transfer(target, self._to_backend(matrix.T), where, target.lower_ind)
                elif has_noise:
                    # Before the first noise update the forward and inverse
                    # unitary layers cancel around the identity map.
                    matrix = self._to_backend(matrix)
                    _gate_transfer(target, matrix, where, target.upper_ind)
                    _gate_transfer(target, matrix, where, target.lower_ind)
            has_noise = True
            mpo = self._compress_layer(target, step)
        return mpo

    def _compress_layer(self, target, step):
        if self._chi is not None and self._block_size is not None and self.n_qubits > 1:
            mpo, _ = _fit_lazy_mpo_target(
                target, chi=self._chi, block_size=min(self._block_size, self.n_qubits),
                cutoff=self._cutoff, cutoff_mode=self._cutoff_mode,
                guess_method="src",
                guess_seed=self._fit_init_seed + step, n_iter=self._n_iter,
            )
            return mpo

        exact = self._chi is None or self.n_qubits == 1
        method = "direct" if exact else self._method
        options = MpoOptimizer._submpo_compress_options(
            method, cutoff=0.0 if exact else self._cutoff, cutoff_mode=self._cutoff_mode,
            max_bond=self._chi, seed=self._fit_init_seed + step,
        )
        if method == "fit-projector":
            options["canonize"] = False
        return run_seeded_quimb(
            options.pop("seed", None), qtn.tensor_network_1d_compress, target,
            method=method, permute_arrays=False, **options,
        )

    def build_mpo(self):
        """Return an owned copy of the mitigation map in the I, X, Y, Z basis."""
        return self._mitigation.copy(deep=True)

    def collect_shots(
        self, initial_mps, shots, *, seed=None, basis_probs=(1 / 3, 1 / 3, 1 / 3),
        mps_chi=32, mps_mode="direct", strategy="auto", batch_size=4096,
        max_branches=None, parallel_workers=1, mps_run_kwargs=None,
    ):
        """Sample trajectories and randomized Pauli readouts before building the map.

        Auto and independent replay work in bounded batches. Coalesced replay
        shares trajectory states across all shots, subject to its branch budget.
        """
        shots = _positive_integer(shots, "shots")
        mps_chi = _positive_integer(mps_chi, "mps_chi")
        batch_size = _positive_integer(batch_size, "batch_size")
        strategy = str(strategy).strip().lower()
        if initial_mps.L != self.n_qubits or any(initial_mps.phys_dim(i) != 2 for i in range(initial_mps.L)):
            raise ValueError("initial_mps must contain one qubit per circuit site.")
        backends = {ar.infer_backend(array) for array in initial_mps.arrays}
        if backends not in ({"numpy"}, {"torch"}):
            raise TypeError("collect_shots expects a dense NumPy or Torch MPS.")
        probs = _basis_probabilities(basis_probs, self.n_qubits)
        simulation_seed, measurement_seed = np.random.SeedSequence(seed).spawn(2)
        simulation_rng = np.random.default_rng(simulation_seed)
        rng = np.random.default_rng(measurement_seed)
        state = initial_mps.copy()
        if backends == {"numpy"}:
            state.apply_to_arrays(lambda array: np.asarray(array, dtype=complex))
        else:
            _patch_backend_string_namespace_key()
            reg_stop_gradient_torch()
            state.apply_to_arrays(lambda array: ar.astype(
                array, "complex64" if ar.get_dtype_name(array) in {"float32", "complex64"} else "complex128"
            ))
        branch_options = {} if max_branches is None else {"max_branches": max_branches}
        if strategy == "coalesced":
            batch_size = shots
        readouts = Counter()
        thresholds = np.cumsum(probs, axis=1)[:, :2]
        for start in range(0, shots, batch_size):
            trajectories = run_trajectory_shots(
                lambda: MpsOptimizer(state, chi=mps_chi, mode=mps_mode),
                self._stream,
                min(batch_size, shots - start),
                seed=int(simulation_rng.integers(2**63)),
                strategy=strategy,
                parallel_workers=parallel_workers,
                run_kwargs=mps_run_kwargs,
                retain="final",
                **branch_options,
            )
            for readout, count in _trajectory_readouts(trajectories, thresholds, rng):
                readouts[readout] += count
        values = np.array(tuple(readouts), dtype=np.int8)
        counts = np.fromiter(readouts.values(), dtype=np.int64)
        return PauliShots(values[:, :self.n_qubits], values[:, self.n_qubits:], counts, probs)

    simulate_shots = collect_shots

    def estimate(self, observables, shots, *, chunk_size=4096):
        """Score Pauli strings against the same raw and mitigated readouts."""
        if not isinstance(shots, PauliShots) or shots.bases.shape[1] != self.n_qubits:
            raise ValueError("shots must be PauliShots for this circuit's qubit count.")
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        strings = (observables,) if isinstance(observables, str) else tuple(observables)
        for string in strings:
            if not isinstance(string, str) or len(string) != self.n_qubits or set(string) - set("IXYZ"):
                raise ValueError("Use one I, X, Y, or Z per qubit in each observable string.")
        total = shots.shots
        estimates = {}
        for string, (raw, mitigated) in self._shot_values(strings, shots, chunk_size):
            raw_mean, raw_error = _shot_stats(raw, shots.counts, total)
            mean, error = _shot_stats(mitigated, shots.counts, total)
            estimates[string] = TemEstimate(raw_mean, mean, raw_error, error)
        return estimates

    def _shot_values(self, strings, shots, chunk_size):
        """Yield raw and mitigated values per readout for each Pauli string."""
        cores = tuple(_ptm_cores(self._mitigation))
        on_numpy = ar.infer_backend(cores[0]) == "numpy"
        to_core = infer_backend_converter_from_sample(cores[0])
        scale = 10.0**self._mitigation.exponent
        n_readouts = len(shots.bases)
        sites = np.arange(self.n_qubits)

        def prepare_chunk(chunk):
            bases = shots.bases[chunk]
            weights = (1 - 2 * shots.bits[chunk]) / shots.basis_probs[sites, bases]
            coefficients = None if on_numpy else to_core(
                weights[..., None] * (bases[..., None] == np.arange(3))
            )
            return bases, weights, coefficients

        # Reuse a device upload across observables when the readouts fit one chunk.
        cached = prepare_chunk(slice(None)) if not on_numpy and n_readouts <= chunk_size else None
        for string in strings:
            axes = tuple("IXYZ".index(pauli) for pauli in string)
            rows = tuple(core[:, :, axis, :] for core, axis in zip(cores, axes, strict=True))
            raw = np.empty(n_readouts)
            mitigated = np.empty(n_readouts)
            for start in range(0, n_readouts, chunk_size):
                chunk = slice(start, start + chunk_size)
                bases, local_weights, coefficients = cached if cached is not None else prepare_chunk(chunk)
                values = to_core(np.ones((len(bases), 1)))
                raw_values = np.ones(len(bases))
                for site, (out, core) in enumerate(zip(axes, rows, strict=True)):
                    next_values = values @ core[:, :, 0]
                    for axis in range(3):
                        if on_numpy:
                            keep = bases[:, site] == axis
                            next_values[keep] += (values[keep] @ core[:, :, axis + 1]) * local_weights[keep, site, None]
                        else:
                            next_values += (values @ core[:, :, axis + 1]) * coefficients[:, site, axis, None]
                    values = next_values
                    if out:
                        raw_values *= np.where(bases[:, site] == out - 1, local_weights[:, site], 0.0)
                raw[chunk] = raw_values
                mitigated[chunk] = ar.to_numpy(ar.do("real", values[:, 0])) * scale
            yield string, (raw, mitigated)
