"""Perfect sampler for tree tensor-network states.

:class:`TreeSampler` is the tree analogue of :class:`pepsy.MpsSampler`: it draws
exact Born samples from a :class:`~pepsy.optimizers.tree.TreeTensorNetwork`
(or the live state of a :class:`~pepsy.optimizers.TreeOptimizer`) with the same
public surface -- :meth:`~TreeSampler.sample`, :meth:`~TreeSampler.sample_batch`,
:meth:`~TreeSampler.sample_arrays`, :meth:`~TreeSampler.amplitudes`,
:meth:`~TreeSampler.probabilities`, and :meth:`~TreeSampler.refresh` -- and the
same batched, environment-caching efficiency.

Algorithm
---------
The state is first put in canonical form with the orthogonality centre on the
root and normalized, so every non-root node is isometric toward its parent
bond.  Sampling then walks the tree depth-first, carrying a per-sample
**reduced density matrix** on the active parent bond.  At each node the
not-yet-visited sibling subtrees telescope to the identity (the isometry
property), so the density transfer stays bounded by the bond dimension squared
-- the exact tree generalisation of the MPS right-environment sweep, where a
tree's several open sibling bonds replace the MPS's single right bond.  All
``n_samples`` share the cached node arrays and are advanced together with
batched ``einsum`` contractions.  The returned probability of each shot is the
exact product of its conditional Born probabilities, so it equals
``|<config|psi>|**2`` for the normalized state.

Symmray states
--------------
Native Symmray trees can be selected with ``backend="symmray"``.  This path
keeps the canonical tree block-sparse and performs exact projected-norm and
amplitude contractions without calling ``to_dense``.  It supports both
ordinary Abelian and fermionic Symmray arrays; source physical charge-sector
maps are retained so sampled integer codes remain interpretable even when
canonicalisation removes an identically-zero sector.  Fermionic results also
carry a :class:`FermionConfigurationEncoding` when their physical sectors are
one of Pepsy's supported spinless or spinful layouts.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import autoray as ar

from .samplers import (
    FermionConfigurationEncoding,
    _backend_array_to_numpy,
    _fermion_symmray_occupations,
    _mps_array_backend,
)

__all__ = [
    "TreeBatchSampleResult",
    "TreeSampleResult",
    "TreeSampler",
]

try:  # threadpoolctl is a NumPy/SciPy transitive dependency; treat as optional.
    from threadpoolctl import ThreadpoolController as _ThreadpoolController

    _THREAD_CONTROLLER = _ThreadpoolController()
except Exception:  # pragma: no cover - threadpoolctl missing
    _THREAD_CONTROLLER = None


def _normalize_tree_sampler_backend(backend):
    """Normalize the small backend vocabulary used by :class:`TreeSampler`."""
    if backend is None:
        return "auto"
    text = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "native": "native",
        "device": "native",
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
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "backend must be one of 'auto', 'native', 'numpy', 'torch', "
            "'cupy', or 'symmray'."
        ) from exc


def _fermion_code_occupations(phys_dim, spinful):
    """Return Symmray's dense-basis code -> occupation table, or ``None``.

    The physical legs of a native fermionic tree densify in Symmray's canonical
    basis order.  For a spinful ``phys_dim=4`` site that order is ``empty, up,
    down, up-down`` (matching ``number_u = diag(0, 1, 0, 1)`` and ``number_d =
    diag(0, 0, 1, 1)``); a spinless ``phys_dim=2`` site is ``empty, occupied``.
    Occupations are returned as ``(n_up, n_down)`` tuples when spinful and
    ``(n,)`` tuples when spinless.  Any other layout returns ``None`` so the
    sampler simply omits an occupation decoder.
    """
    if spinful and phys_dim == 4:
        return ((0, 0), (1, 0), (0, 1), (1, 1))
    if not spinful and phys_dim == 2:
        return ((0,), (1,))
    return None


@dataclass
class TreeSampleResult:
    """Container for tree-tensor-network samples.

    Attributes
    ----------
    configs : list[list[int]]
        Each entry is a list of length ``nqubits`` with the sampled
        computational-basis index (0 or 1) for each qubit ``0..nqubits-1``.
    probs : list[float]
        Born probability ``|<config|psi>|**2`` for each sample.
    nqubits : int
        Number of physical qubit sites in the tree.
    """

    configs: list[list[int]]
    probs: list[float]
    nqubits: int
    configuration_encoding: FermionConfigurationEncoding | None = None

    def __len__(self):
        return len(self.configs)

    def occupations(self, *, to_numpy: bool = False):
        """Decode fermionic physical codes into on-site occupations.

        Requires a fermionic sampler.  Spinful results have shape
        ``(n_samples, nqubits, 2)`` in ``(n_up, n_down)`` order; spinless
        results have shape ``(n_samples, nqubits)``.
        """
        if self.configuration_encoding is None:
            raise ValueError(
                "This result has no fermion configuration encoding; sample from "
                "a fermionic tree state (optionally pass fermion=... to "
                "TreeSampler)."
            )
        return self.configuration_encoding.decode(self.configs, to_numpy=to_numpy)

    def magnetizations(self) -> np.ndarray:
        """Per-sample magnetization ``(1 / n) * sum_i (1 - 2 * bit_i)``."""
        n = self.nqubits
        return np.array([
            np.sum(1 - 2 * np.array(c)) / n for c in self.configs
        ])


@dataclass
class TreeBatchSampleResult:
    """Batched tree-tensor-network samples.

    Attributes
    ----------
    configs : np.ndarray
        Integer array with shape ``(n_samples, nqubits)`` whose column ``q``
        holds the sampled bit for qubit ``q``.
    probs : np.ndarray
        Born probabilities for ``configs`` with shape ``(n_samples,)``.
    nqubits : int
        Number of physical qubit sites in the tree.
    """

    configs: object
    probs: object
    nqubits: int
    configuration_encoding: FermionConfigurationEncoding | None = None
    backend: str = "numpy"

    def __len__(self):
        return int(self.configs.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of sampled configurations."""
        return len(self)

    def to_numpy(self) -> "TreeBatchSampleResult":
        """Return a plain NumPy copy of this batched result."""
        return TreeBatchSampleResult(
            configs=_backend_array_to_numpy(self.configs),
            probs=_backend_array_to_numpy(self.probs),
            nqubits=self.nqubits,
            configuration_encoding=self.configuration_encoding,
            backend="numpy",
        )

    def occupations(self, *, to_numpy: bool = False):
        """Decode fermionic physical codes into on-site occupations.

        Requires a fermionic sampler.  Spinful results have shape
        ``(n_samples, nqubits, 2)`` in ``(n_up, n_down)`` order; spinless
        results have shape ``(n_samples, nqubits)``.
        """
        if self.configuration_encoding is None:
            raise ValueError(
                "This batch has no fermion configuration encoding; sample from "
                "a fermionic tree state (optionally pass fermion=... to "
                "TreeSampler)."
            )
        return self.configuration_encoding.decode(self.configs, to_numpy=to_numpy)

    def configs_list(self) -> list[list[int]]:
        """Return configurations as Python ``list[list[int]]``."""
        configs = _backend_array_to_numpy(self.configs)
        return [[int(value) for value in config] for config in configs]

    def magnetizations(self, *, to_numpy: bool = True):
        """Per-sample magnetization ``(1 / n) * sum_i (1 - 2 * bit_i)``."""
        if self.backend == "torch":
            import torch

            configs = self.configs.to(dtype=torch.float64)
            values = (1.0 - 2.0 * configs).sum(dim=1) / float(self.nqubits)
        elif self.backend == "cupy":
            import cupy as cp

            configs = self.configs.astype(cp.float64, copy=False)
            values = (1.0 - 2.0 * configs).sum(axis=1) / float(self.nqubits)
        else:
            configs = np.asarray(self.configs, dtype=float)
            values = (1 - 2 * configs).sum(axis=1) / float(self.nqubits)
        return _backend_array_to_numpy(values) if to_numpy else values

    def to_sample_result(self) -> TreeSampleResult:
        """Convert to the list-based :class:`TreeSampleResult`."""
        batch = self.to_numpy()
        return TreeSampleResult(
            configs=batch.configs_list(),
            probs=[float(p) for p in batch.probs],
            nqubits=self.nqubits,
            configuration_encoding=self.configuration_encoding,
        )


class TreeSampler:
    """Draw exact Born samples from a tree tensor-network state.

    Parameters
    ----------
    state : TreeTensorNetwork or TreeOptimizer
        The tree state to sample from.  A :class:`~pepsy.TreeOptimizer` is
        accepted directly (its live :attr:`~pepsy.TreeOptimizer.tn` is used);
        the source object is never mutated.
    seed : None, int, or np.random.Generator, optional
        Seed (or generator) for the sampler's persistent random generator,
        used when :meth:`sample` is called without a per-call ``seed``.
    threads : int or None, default=1
        BLAS/OpenMP thread cap applied around the batched contractions.  Tree
        node arrays are small (bounded by the bond dimension), so a single
        thread is typically fastest; pass ``None`` to leave the ambient thread
        count untouched.
    backend : {"auto", "native", "numpy", "torch", "cupy", "symmray"}, default="auto"
        Backend used for cached node arrays and batched contractions. ``auto``
        preserves the existing dense compatibility path for Symmray trees;
        ``native`` or ``symmray`` selects the block-sparse Symmray path.
        Explicit Torch or CuPy requests require the live state to already use
        that dense backend.
    fermion : pepsy.tensors.Fermion, optional
        Fermionic physical-space convention.  It is optional: a native Symmray
        fermionic tree is detected automatically and its physical-code
        occupation decoder is attached to the sample results when supported.
        Supplying a ``fermion`` pins the recorded ``symmetry``/``spinful``
        labels.

    Notes
    -----
    The node arrays are extracted once (in canonical, normalized form) and
    cached.  Call :meth:`refresh` after the source state changes; otherwise the
    sampler keeps representing its previously captured tensor data.
    """

    def __init__(
        self,
        state,
        *,
        seed=None,
        threads: int | None = 1,
        backend="auto",
        fermion=None,
    ):
        self._rng = np.random.default_rng(seed)
        self.threads = None if threads is None else int(threads)
        self.backend = _normalize_tree_sampler_backend(backend)
        self.resolved_backend = None
        self._fermion = fermion
        self._device = None

        # Geometry / cached canonical arrays (populated by refresh).
        self._nqubits = None
        self._root = None
        self._children = None
        self._qubit_of_node = None
        self._arrays = None
        self._fermionic = False
        self._configuration_encoding = None
        self._physical_code_maps = None
        self._symmray_state = None

        self.refresh(state)

    # -- setup ---------------------------------------------------------------

    def _thread_ctx(self):
        """Context manager capping BLAS/OpenMP threads for the batched ops."""
        if _THREAD_CONTROLLER is not None and self.threads is not None:
            return _THREAD_CONTROLLER.limit(limits=self.threads)
        return contextlib.nullcontext()

    @staticmethod
    def _dense_data(data):
        """Return a dense view while retaining its native array backend."""
        return data.to_dense() if hasattr(data, "to_dense") else data

    @staticmethod
    def _state_ttn(state):
        """Return the tree state owned by a TTN or TreeOptimizer."""
        tn = getattr(state, "tn", None)
        if tn is None:
            tn = state
        if not hasattr(tn, "plan") or not hasattr(tn, "node_tensor"):
            raise TypeError(
                "TreeSampler expects a TreeTensorNetwork or a TreeOptimizer; "
                f"got {type(state).__name__}."
            )
        return tn

    @staticmethod
    def _is_symmray_ttn(tn):
        """Check that a TTN uses Symmray data consistently on every node."""
        backends = {_mps_array_backend(tensor.data) for tensor in tn.tensors}
        if "symmray" not in backends:
            return False
        if backends != {"symmray"}:
            raise ValueError(
                "TreeSampler requires either all dense or all Symmray tree "
                f"tensors; got mixed backends {sorted(backends)!r}."
            )
        return True

    def _resolve_backend(self, tn):
        """Resolve the requested backend against the live tree state."""
        if self._is_symmray_ttn(tn):
            if self.backend in {"native", "symmray"}:
                return "symmray"
            if self.backend in {"auto", "numpy"}:
                # Keep the established default fast batched path. Native
                # block-sparse sampling is explicit because its tree branch
                # contractions currently trade throughput for generality.
                return "numpy"
            raise ValueError(
                f"TreeSampler backend={self.backend!r} requested for a "
                "Symmray tree. Use backend='symmray' (or 'native') for the "
                "block-sparse path, or backend='numpy' for dense sampling."
            )

        data = tn.node_tensor(tn.root).data
        source = _mps_array_backend(data)
        if source not in {"numpy", "torch", "cupy"}:
            source = "numpy"

        if self.backend in {"auto", "native"}:
            return source
        if self.backend == "numpy":
            return "numpy"
        if source != self.backend:
            raise ValueError(
                f"TreeSampler backend={self.backend!r} requires a live "
                f"{self.backend} tree, but the state uses {source!r}. "
                "Use TreeOptimizer.to_backend(...) for operators and move "
                "the live tree tensors to the requested backend first."
            )
        return self.backend

    def _xp(self):
        if self.resolved_backend == "torch":
            import torch

            return torch
        if self.resolved_backend == "cupy":
            import cupy as cp

            return cp
        return np

    def _as_backend(self, array, *, dtype=None):
        """Convert a host/foreign array onto the sampler backend and device."""
        if self.resolved_backend == "torch":
            import torch

            if torch.is_tensor(array):
                kwargs = {"device": self._device}
                if dtype is not None:
                    kwargs["dtype"] = dtype
                return array.to(**kwargs)
            return torch.as_tensor(
                _backend_array_to_numpy(array), dtype=dtype, device=self._device
            )
        if self.resolved_backend == "cupy":
            import cupy as cp

            if isinstance(array, cp.ndarray):
                return array.astype(dtype, copy=False) if dtype is not None else array
            out = cp.asarray(array)
            return out.astype(dtype, copy=False) if dtype is not None else out
        out = np.asarray(_backend_array_to_numpy(array))
        return out.astype(dtype, copy=False) if dtype is not None else out

    def _zeros(self, shape, *, dtype):
        if self.resolved_backend == "torch":
            import torch

            return torch.zeros(shape, dtype=dtype, device=self._device)
        return self._xp().zeros(shape, dtype=dtype)

    def _ones(self, shape, *, dtype):
        if self.resolved_backend == "torch":
            import torch

            return torch.ones(shape, dtype=dtype, device=self._device)
        return self._xp().ones(shape, dtype=dtype)

    def _arange(self, n):
        if self.resolved_backend == "torch":
            import torch

            return torch.arange(n, dtype=torch.int64, device=self._device)
        return self._xp().arange(n, dtype=np.int64)

    def _sum(self, value, axis=None, *, keepdims=False):
        if self.resolved_backend == "torch":
            import torch

            return torch.sum(value, dim=axis, keepdim=keepdims)
        return self._xp().sum(value, axis=axis, keepdims=keepdims)

    def _einsum(self, equation, *operands):
        if self.resolved_backend == "torch":
            import torch

            return torch.einsum(equation, operands)
        return self._xp().einsum(equation, *operands)

    def _tensordot(self, left, right, axes):
        if self.resolved_backend == "torch":
            import torch

            return torch.tensordot(left, right, dims=axes)
        return self._xp().tensordot(left, right, axes=axes)

    def _moveaxis(self, value, source, destination):
        if self.resolved_backend == "torch":
            import torch

            return torch.movedim(value, source, destination)
        return self._xp().moveaxis(value, source, destination)

    def _cumsum(self, value, axis):
        if self.resolved_backend == "torch":
            import torch

            return torch.cumsum(value, dim=axis)
        return self._xp().cumsum(value, axis=axis)

    def _where(self, condition, left, right):
        if self.resolved_backend == "torch":
            import torch

            return torch.where(condition, left, right)
        return self._xp().where(condition, left, right)

    def _clip_nonnegative(self, value):
        if self.resolved_backend == "torch":
            return value.clamp_min(0.0)
        return self._xp().clip(value, 0.0, None)

    def _clip_max(self, value, maximum):
        if self.resolved_backend == "torch":
            return value.clamp_max(maximum)
        return self._xp().minimum(value, maximum)

    def _as_int64(self, value):
        if self.resolved_backend == "torch":
            import torch

            return value.to(dtype=torch.int64)
        return value.astype(np.int64, copy=False)

    def _broadcast_to(self, value, shape):
        """Broadcast ``value`` without changing its native backend."""
        if self.resolved_backend == "torch":
            import torch

            return torch.broadcast_to(value, tuple(shape))
        return self._xp().broadcast_to(value, tuple(shape))

    @staticmethod
    def _resolve_ttn(state):
        """Return a canonical, normalized :class:`TreeTensorNetwork` copy."""
        tn = TreeSampler._state_ttn(state)
        tn = tn.copy()
        # Put the orthogonality centre on the root: every other node becomes
        # isometric toward its parent bond, which is what the sampling sweep
        # relies on to collapse unvisited subtrees to the identity.
        tn.canonize_around_node_(tn.root)
        return tn

    def refresh(self, state=None):
        """Re-capture cached arrays from ``state`` (or the previous source).

        Returns
        -------
        TreeSampler
            This sampler, with its canonical node arrays rebuilt.
        """
        if state is None:
            state = self._source
        if state is None:
            raise ValueError("refresh requires a tree state before initialization.")
        self._source = state

        with self._thread_ctx():
            source_tn = self._state_ttn(state)
            tn = self._resolve_ttn(state)
            self.resolved_backend = self._resolve_backend(tn)
            if self.resolved_backend == "symmray":
                self._extract_symmray(tn, source_tn)
            else:
                self._extract_arrays(tn, source_tn)
        return self

    @staticmethod
    def _symmray_code_metadata(index):
        """Expand a Symmray physical index into ``(charge, sector)`` codes."""
        chargemap = getattr(index, "chargemap", None)
        if chargemap is None:
            raise TypeError("Symmray physical indices must expose a chargemap.")
        metadata = []
        for charge, size in chargemap.items():
            for sector_offset in range(int(size)):
                metadata.append((charge, sector_offset))
        return tuple(metadata)

    @classmethod
    def _symmray_metadata(cls, source_tn, canonical_tn):
        """Retain source physical codes while indexing canonical tensors."""
        source_maps = []
        local_to_source = []
        source_to_local = []
        for qubit in range(int(canonical_tn.plan.n)):
            source_tensor = source_tn.node_tensor(source_tn.node_of_qubit(qubit))
            canonical_tensor = canonical_tn.node_tensor(
                canonical_tn.node_of_qubit(qubit)
            )
            source_phys = source_tensor.data.indices[
                source_tensor.inds.index(source_tn.site_ind(qubit))
            ]
            canonical_phys = canonical_tensor.data.indices[
                canonical_tensor.inds.index(canonical_tn.site_ind(qubit))
            ]
            source_metadata = cls._symmray_code_metadata(source_phys)
            canonical_metadata = cls._symmray_code_metadata(canonical_phys)
            source_lookup = {
                metadata: code
                for code, metadata in enumerate(source_metadata)
            }
            try:
                code_map = tuple(source_lookup[metadata] for metadata in canonical_metadata)
            except KeyError as exc:
                raise ValueError(
                    "Canonical Symmray tree changed a physical charge sector "
                    f"at site {qubit}: {exc.args[0]!r}."
                ) from exc
            source_maps.append(dict(enumerate(source_metadata)))
            local_to_source.append(code_map)
            source_to_local.append({source: local for local, source in enumerate(code_map)})

        data = source_tn.node_tensor(source_tn.root).data
        block_backends = {
            str(getattr(tensor.data, "backend", "unknown"))
            for tensor in canonical_tn.tensors
        }
        if len(block_backends) != 1:
            raise ValueError(
                "Symmray tree tensors must use one common underlying block "
                f"backend; got {sorted(block_backends)!r}."
            )
        array_backend = next(iter(block_backends))
        if array_backend not in {"numpy", "torch", "cupy"}:
            raise ValueError(
                "TreeSampler Symmray sampling supports NumPy, Torch, or CuPy "
                f"blocks, not {array_backend!r}."
            )
        return (
            tuple(source_maps),
            tuple(local_to_source),
            tuple(source_to_local),
            array_backend,
            data.get_any_array(),
        )

    @staticmethod
    def _symmray_scalar(value):
        """Extract a scalar without densifying a Symmray array."""
        if hasattr(value, "data") and not np.isscalar(value):
            value = value.data
        blocks = getattr(value, "blocks", None)
        if isinstance(blocks, dict) and not blocks:
            return 0.0
        if hasattr(value, "get_scalar_element"):
            return value.phase_sync().get_scalar_element()
        return value

    @classmethod
    def _symmray_norm_squared(cls, tn):
        """Return a full native Symmray norm without a dense conversion."""
        value = (tn.H | tn).contract(all, optimize="auto")
        value = cls._symmray_scalar(value)
        return float(np.real(ar.to_numpy(value)))

    def _symmray_output(self, values, *, dtype):
        """Stack host sampling results on the Symmray block backend."""
        state = self._symmray_state
        backend = state["array_backend"]
        template = state["template"]
        if backend == "torch":
            import torch

            return torch.as_tensor(values, dtype=dtype, device=template.device)
        if backend == "cupy":
            import cupy as cp

            return cp.asarray(values, dtype=dtype)
        return np.asarray(values, dtype=dtype)

    def _extract_symmray(self, tn, source_tn):
        """Prepare a native Symmray tree and retain its physical code maps."""
        (
            physical_code_maps,
            local_to_source,
            source_to_local,
            array_backend,
            template,
        ) = self._symmray_metadata(source_tn, tn)
        norm_squared = self._symmray_norm_squared(tn)
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ValueError("Symmray tree state has a zero or non-finite norm.")
        root_tensor = tn.node_tensor(tn.root)
        root_tensor.modify(data=root_tensor.data / math.sqrt(norm_squared))
        if hasattr(tn, "_invalidate_norm_cache"):
            tn._invalidate_norm_cache()

        plan = tn.plan
        self._nqubits = int(plan.n)
        self._root = plan.root
        self._children = {nid: tuple(plan.children[nid]) for nid in plan.children}
        self._qubit_of_node = dict(plan.qubit_of_node)
        self._node_of_qubit = {
            int(qubit): int(node)
            for node, qubit in plan.qubit_of_node.items()
        }
        self._edges = tuple(
            (int(parent), int(child))
            for child, parent in sorted(
                plan.parent.items(),
                key=lambda item: (int(item[0]), int(item[1])),
            )
        )
        preorder = []
        stack = [self._root]
        while stack:
            nid = stack.pop()
            preorder.append(nid)
            stack.extend(reversed(self._children[nid]))
        self._sampling_qubits = tuple(
            int(self._qubit_of_node[nid])
            for nid in preorder
            if nid in self._qubit_of_node
        )
        self._arrays = None
        self._fermionic = bool(getattr(tn, "fermionic", False))
        self._physical_code_maps = physical_code_maps
        self._symmray_state = {
            "tn": tn,
            "physical_code_maps": physical_code_maps,
            "local_to_source": local_to_source,
            "source_to_local": source_to_local,
            "array_backend": array_backend,
            "template": template,
        }
        self._configuration_encoding = (
            self._build_configuration_encoding(
                tn,
                None,
                self._qubit_of_node,
                code_metadata=physical_code_maps,
            )
            if self._fermionic
            else None
        )

    def _extract_arrays(self, tn, source_tn=None):
        """Extract per-node dense arrays with a canonical axis order.

        Every array starts with its parent bond, with the root receiving a
        dummy parent bond of size 1. A physical site's axis comes next, followed
        by its child bonds. Thus a leaf has ``(parent_bond, phys)``, a
        virtual-only internal node has ``(parent_bond, child_0, ...)``, and a
        physical root has ``(dummy_parent, phys, child_0, ...)``. The root array
        is L2-normalized in place, which normalizes the whole state because
        every other tensor is isometric in this canonical form.

        Bond indices are resolved by intersecting adjacent node tensors rather
        than by the deterministic ``_tb{lo}_{hi}`` names, because gate threading
        and canonicalisation replace those with fresh ``quimb`` uuids.
        """
        plan = tn.plan
        root = plan.root
        children = {nid: tuple(plan.children[nid]) for nid in plan.children}
        qubit_of_node = dict(plan.qubit_of_node)

        # Native Symmray fermionic trees keep block-sparse arrays; densify them
        # once.  A graded-canonical tensor is also plain-isometric (its exchange
        # signs cancel in the ket-with-bra reduced-density contraction), so the
        # dense sweep below reproduces the exact fermionic Born probabilities.
        fermionic = bool(getattr(tn, "fermionic", False))
        self._device = None

        def to_arr(data):
            data = self._dense_data(data) if fermionic else data
            arr = self._as_backend(data)
            if self.resolved_backend == "torch" and self._device is None:
                self._device = arr.device
            return arr

        def bond_between(a, b):
            shared = set(tn.node_tensor(a).inds) & set(tn.node_tensor(b).inds)
            if len(shared) != 1:
                raise ValueError(
                    f"nodes {a} and {b} must share exactly one bond; "
                    f"found {sorted(shared)}."
                )
            return next(iter(shared))

        arrays = {}
        for nid, ch in children.items():
            t = tn.node_tensor(nid)
            is_root = nid == root
            parent = plan.parent.get(nid)
            ordered_inds = []
            if not is_root:
                ordered_inds.append(bond_between(nid, parent))
            q = qubit_of_node.get(nid)
            if q is not None:
                ordered_inds.append(tn.site_ind(q))
            ordered_inds.extend(bond_between(nid, child) for child in ch)
            arr = to_arr(t.transpose(*ordered_inds).data)
            if is_root:
                arr = arr.reshape((1,) + arr.shape)
            arrays[nid] = arr

        # Normalize via the root array (state is canonical with centre = root).
        root_arr = arrays[root]
        nrm = math.sqrt(
            float(ar.to_numpy(self._sum(self._xp().abs(root_arr) ** 2)))
        )
        if nrm > 0:
            arrays[root] = root_arr / nrm

        self._nqubits = int(plan.n)
        self._root = root
        self._children = children
        self._qubit_of_node = qubit_of_node
        self._node_of_qubit = {
            int(qubit): int(node)
            for node, qubit in qubit_of_node.items()
        }
        self._edges = tuple(
            (int(parent), int(child))
            for child, parent in sorted(
                plan.parent.items(),
                key=lambda item: (int(item[0]), int(item[1])),
            )
        )
        preorder = []
        stack = [root]
        while stack:
            nid = stack.pop()
            preorder.append(nid)
            stack.extend(reversed(children[nid]))
        self._sampling_qubits = tuple(
            int(qubit_of_node[nid])
            for nid in preorder
            if nid in qubit_of_node
        )
        self._arrays = arrays
        self._fermionic = fermionic
        self._symmray_state = None
        if source_tn is not None and self._is_symmray_ttn(source_tn):
            (
                self._physical_code_maps,
                _local_to_source,
                _source_to_local,
                _array_backend,
                _template,
            ) = self._symmray_metadata(source_tn, tn)
        else:
            self._physical_code_maps = None
        self._configuration_encoding = (
            self._build_configuration_encoding(
                tn,
                arrays,
                qubit_of_node,
                code_metadata=self._physical_code_maps,
            )
            if fermionic
            else None
        )

    def _selected_node_array(self, nid, configs):
        """Select the sampled physical value at one node, in batch form."""
        arr = self._arrays[nid]
        qubit = self._qubit_of_node.get(nid)
        if qubit is not None:
            return self._moveaxis(arr[:, configs[:, qubit], ...], 1, 0)
        return self._broadcast_to(
            arr,
            (int(configs.shape[0]),) + tuple(arr.shape),
        )

    def _contract_child_axis(self, value, vector, axis):
        """Contract one batched tree bond while keeping the first leg open."""
        value = self._moveaxis(value, axis, 2)
        batch = int(value.shape[0])
        target = int(value.shape[1])
        child = int(value.shape[2])
        rest = tuple(value.shape[3:])
        flat = value.reshape(batch, target, child, -1)
        contracted = self._einsum("BtcF,Bc->BtF", flat, vector)
        return contracted.reshape((batch, target) + rest)

    def _message_pass(self, configs):
        """Build upward amplitudes and downward environments for a batch."""
        batch = int(configs.shape[0])
        preorder = []
        stack = [self._root]
        while stack:
            nid = stack.pop()
            preorder.append(nid)
            stack.extend(reversed(self._children[nid]))

        upward = {}
        for nid in reversed(preorder):
            value = self._selected_node_array(nid, configs)
            for child in self._children[nid]:
                value = self._contract_child_axis(
                    value, upward[child], axis=2
                )
            upward[nid] = value.reshape(batch, value.shape[1])

        root_dtype = self._arrays[self._root].dtype
        downward = {
            self._root: self._ones((batch, 1), dtype=root_dtype),
        }
        for nid in preorder:
            children = self._children[nid]
            if not children:
                continue
            value = self._selected_node_array(nid, configs)
            flat = value.reshape(batch, value.shape[1], -1)
            weighted = self._einsum("Bp,BpF->BF", downward[nid], flat)
            child_shape = tuple(value.shape[2:])
            weighted = weighted.reshape((batch,) + child_shape)
            for child_index, child in enumerate(children):
                value_for_child = self._moveaxis(
                    weighted, child_index + 1, 1
                )
                remaining = list(children)
                remaining.pop(child_index)
                for sibling in tuple(remaining):
                    sibling_axis = 2 + remaining.index(sibling)
                    value_for_child = self._contract_child_axis(
                        value_for_child,
                        upward[sibling],
                        axis=sibling_axis,
                    )
                    remaining.remove(sibling)
                downward[child] = value_for_child.reshape(
                    batch, value_for_child.shape[1]
                )
        return upward, downward

    def single_site_flip_amplitude_ratios(
        self,
        configs,
        *,
        to_numpy: bool = True,
    ):
        """Return ``psi(x with site flipped) / psi(x)`` for binary trees.

        The upward/downward message pass contracts the tree once per batch and
        then evaluates each leaf flip from its local environment. This avoids
        constructing a separate full-tree amplitude contraction for every
        site, which is the fallback used by generic samplers.
        """
        configs = self._check_configs(configs)
        if self.resolved_backend == "symmray":
            state = self._symmray_state
            if any(
                len(state["local_to_source"][qubit]) != 2
                for qubit in range(self._nqubits)
            ):
                raise NotImplementedError(
                    "single-site flip ratios require binary physical dimensions."
                )
            flipped = np.asarray(configs, dtype=np.int64).copy()
            for qubit in range(self._nqubits):
                source_to_local = state["source_to_local"][qubit]
                local_to_source = state["local_to_source"][qubit]
                try:
                    flipped[:, qubit] = np.asarray([
                        local_to_source[1 - source_to_local[int(code)]]
                        for code in flipped[:, qubit]
                    ])
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(
                        f"configs contain invalid physical index for site {qubit}."
                    ) from exc
            ratios = (
                self._symmray_amplitudes(flipped)
                / self._symmray_amplitudes(configs)
            )
            return _backend_array_to_numpy(ratios) if to_numpy else ratios
        if any(
            int(self._arrays[node].shape[1]) != 2
            for node in self._node_of_qubit.values()
        ):
            raise NotImplementedError(
                "single-site flip ratios require binary physical dimensions."
            )
        with self._thread_ctx():
            upward, downward = self._message_pass(configs)
            amplitudes = upward[self._root][:, 0]
            ratios = self._zeros(
                (int(configs.shape[0]), self._nqubits),
                dtype=amplitudes.dtype,
            )
            for qubit in range(self._nqubits):
                nid = self._node_of_qubit[qubit]
                arr = self._arrays[nid]
                flipped = self._moveaxis(
                    arr[:, 1 - configs[:, qubit], ...], 1, 0
                )
                for child in self._children[nid]:
                    flipped = self._contract_child_axis(
                        flipped, upward[child], axis=2
                    )
                flipped_amplitudes = self._sum(
                    downward[nid] * flipped.reshape(
                        int(configs.shape[0]), flipped.shape[1]
                    ),
                    axis=1,
                )
                ratios[:, qubit] = flipped_amplitudes / amplitudes
        return _backend_array_to_numpy(ratios) if to_numpy else ratios

    def tree_edge_entropies(self, *, method="svd", return_edges=False):
        """Measure edge entropies from the cached canonical sampler tree."""
        if self.resolved_backend == "symmray":
            return self._symmray_state["tn"].tree_edge_entropies(
                method=method,
                return_edges=return_edges,
            )
        method = str(method).strip().lower().replace("_", ":")
        if method not in {"svd", "eig", "svd:eig"}:
            raise ValueError(
                "tree entropy method must be 'svd', 'eig', or 'svd:eig'."
            )
        values = []
        for _parent, child in self._edges:
            matrix = self._arrays[child].reshape(
                int(self._arrays[child].shape[0]), -1
            )
            if method in {"eig", "svd:eig"}:
                if self.resolved_backend == "torch":
                    import torch

                    adjoint = matrix.conj().transpose(-2, -1)
                    gram = (
                        matrix @ adjoint
                        if matrix.shape[0] <= matrix.shape[1]
                        else adjoint @ matrix
                    )
                    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
                    singular_values = torch.sqrt(eigenvalues)
                else:
                    xp = self._xp()
                    adjoint = matrix.conj().T
                    gram = (
                        xp.matmul(matrix, adjoint)
                        if matrix.shape[0] <= matrix.shape[1]
                        else xp.matmul(adjoint, matrix)
                    )
                    eigenvalues = xp.linalg.eigvalsh(gram)
                    singular_values = xp.sqrt(xp.clip(eigenvalues, 0.0, None))
            elif self.resolved_backend == "torch":
                import torch

                singular_values = torch.linalg.svdvals(matrix)
            else:
                singular_values = self._xp().linalg.svd(
                    matrix, full_matrices=False, compute_uv=False
                )
            weights = self._xp().abs(singular_values) ** 2
            total = self._sum(weights)
            safe_total = self._where(
                total > 0.0,
                total,
                self._ones((), dtype=weights.dtype),
            )
            probabilities = weights / safe_total
            safe = self._where(
                probabilities > 0.0,
                probabilities,
                self._ones(probabilities.shape, dtype=probabilities.dtype),
            )
            values.append(-self._sum(probabilities * self._xp().log2(safe)))
        if values:
            if self.resolved_backend == "torch":
                import torch

                entropies = torch.stack(values)
            else:
                entropies = self._xp().stack(values)
            entropies = _backend_array_to_numpy(entropies)
        else:
            entropies = np.empty(0, dtype=float)
        entropies = np.asarray(entropies, dtype=float)
        if return_edges:
            return entropies, self._edges
        return entropies

    def _build_configuration_encoding(
        self, tn, arrays, qubit_of_node, *, code_metadata=None
    ):
        """Build the physical-code occupation decoder for a fermionic tree."""
        if not qubit_of_node:
            return None
        first_node = next(iter(qubit_of_node))
        phys_dim = (
            len(code_metadata[0])
            if code_metadata is not None
            else int(arrays[first_node].shape[1])
        )
        fermion = self._fermion
        if fermion is not None and hasattr(fermion, "spinful"):
            spinful = bool(fermion.spinful)
        else:
            spinful = phys_dim == 4
        symmetry = getattr(fermion, "symmetry", None)
        if symmetry is None:
            symmetry = getattr(tn, "symmetry", None)
        symmetry = str(symmetry) if symmetry is not None else "U1U1"
        if code_metadata is not None:
            tables = []
            convention = SimpleNamespace(symmetry=symmetry, spinful=spinful)
            try:
                for code_map in code_metadata:
                    tables.append(tuple(
                        _fermion_symmray_occupations(charge, offset, convention)
                        for charge, offset in code_map.values()
                    ))
            except (TypeError, ValueError, AttributeError):
                return None
            if not tables or any(not table for table in tables):
                return None
        else:
            table = _fermion_code_occupations(phys_dim, spinful)
            if table is None:
                return None
            tables = [tuple(table) for _ in range(self._nqubits)]
        return FermionConfigurationEncoding(
            symmetry=symmetry,
            spinful=spinful,
            code_to_occupations=tuple(tables),
        )

    @property
    def physical_code_maps(self):
        """Return Symmray physical-code to ``(charge, sector)`` maps.

        The maps are ``None`` for dense trees.  For Symmray trees they describe
        the source physical basis, while sampling and evaluation continue to
        use the canonical private copy internally.
        """
        if self._physical_code_maps is None:
            return None
        return tuple(dict(code_map) for code_map in self._physical_code_maps)

    # -- native Symmray path -----------------------------------------------

    def _symmray_project(self, tn, selections):
        """Return a private Symmray tree with physical codes selected."""
        branch = tn.copy()
        branch.isel_({
            branch.site_ind(int(qubit)): int(code)
            for qubit, code in selections.items()
        })
        return branch

    def _symmray_branch_norm(self, tn):
        """Return a selected branch norm without converting it to dense."""
        return self._symmray_norm_squared(tn)

    def _symmray_amplitude_one(self, config):
        """Contract one physical configuration through native Symmray."""
        state = self._symmray_state
        selections = {}
        for qubit, source_code in enumerate(config):
            try:
                selections[qubit] = state["source_to_local"][qubit][int(source_code)]
            except (KeyError, IndexError, TypeError):
                return 0.0
        branch = self._symmray_project(state["tn"], selections)
        value = branch.contract(all, optimize="auto")
        return self._symmray_scalar(value)

    def _symmray_probability_one(self, config):
        """Evaluate one normalized Born probability natively."""
        state = self._symmray_state
        selections = {}
        for qubit, source_code in enumerate(config):
            try:
                selections[qubit] = state["source_to_local"][qubit][int(source_code)]
            except (KeyError, IndexError, TypeError):
                return 0.0
        branch = self._symmray_project(state["tn"], selections)
        return self._symmray_branch_norm(branch)

    def _symmray_amplitudes(self, configs):
        values = [self._symmray_amplitude_one(config) for config in configs]
        template = self._symmray_state["template"]
        if self._symmray_state["array_backend"] in {"torch", "cupy"}:
            dtype = template.dtype
        else:
            dtype = np.result_type(ar.to_numpy(template), complex)
        return self._symmray_output(values, dtype=dtype)

    def _symmray_probabilities(self, configs):
        values = [self._symmray_probability_one(config) for config in configs]
        template = self._symmray_state["template"]
        if self._symmray_state["array_backend"] in {"torch", "cupy"}:
            dtype = template.real.dtype
        else:
            dtype = np.asarray(ar.to_numpy(template)).real.dtype
        return self._symmray_output(values, dtype=dtype)

    def _symmray_sample_arrays(self, n_samples, rng):
        """Sample a general Symmray tree by exact native projected norms.

        A tree has no unique left-to-right prefix boundary.  The canonical
        depth-first physical order gives a simple, fully general fallback:
        each candidate is a private native projection of the already-selected
        tree and its norm supplies the conditional Born probability.  This
        keeps all state contractions block-sparse and is intentionally
        separate from the high-throughput dense batched sweep.
        """
        state = self._symmray_state
        configs = np.empty((int(n_samples), self._nqubits), dtype=np.int64)
        probabilities = np.empty(int(n_samples), dtype=float)
        local_codes = {
            qubit: tuple(range(len(state["local_to_source"][qubit])))
            for qubit in range(self._nqubits)
        }
        for sample in range(int(n_samples)):
            branch = state["tn"]
            branch_norm = 1.0
            for qubit in self._sampling_qubits:
                candidates = []
                weights = []
                for local_code in local_codes[qubit]:
                    candidate = self._symmray_project(
                        branch,
                        {qubit: local_code},
                    )
                    candidates.append(candidate)
                    weights.append(self._symmray_branch_norm(candidate))
                weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
                total = float(weights.sum())
                if not np.isfinite(total) or total <= 0.0:
                    raise ValueError(
                        "Symmray tree sampling reached a zero or non-finite "
                        "conditional norm."
                    )
                probs = weights / total
                choice = int(rng.choice(len(probs), p=probs))
                local_code = local_codes[qubit][choice]
                branch = candidates[choice]
                branch_norm = float(weights[choice])
                configs[sample, qubit] = state["local_to_source"][qubit][local_code]
            probabilities[sample] = branch_norm
        return configs, probabilities

    @property
    def nqubits(self) -> int:
        """Number of physical qubit sites in the tree."""
        return self._nqubits

    # -- sampling ------------------------------------------------------------

    def _sample_arrays(self, n_samples, rng):
        """Batched perfect sampling; returns backend-native arrays."""
        B = int(n_samples)
        arrays = self._arrays
        children = self._children
        qubit_of_node = self._qubit_of_node
        if self.resolved_backend == "torch":
            import torch

            int_dtype = torch.int64
            prob_dtype = torch.float64
        elif self.resolved_backend == "cupy":
            import cupy as cp

            int_dtype = cp.int64
            prob_dtype = cp.float64
        else:
            int_dtype = np.int64
            prob_dtype = np.float64
        configs = self._zeros((B, self._nqubits), dtype=int_dtype)
        prob = self._ones(B, dtype=prob_dtype)
        batch = self._arange(B)
        # Draw all uniforms in one host call and, for accelerator backends,
        # one host-to-device transfer. The previous per-site transfer made a
        # large tree perform one small synchronization/copy per physical site.
        physical_draws = self._as_backend(
            rng.random((len(qubit_of_node), B)), dtype=prob_dtype
        )
        draw_index = 0

        def visit(nid, rho):
            nonlocal draw_index
            # rho: (B, d_par, d_par) reduced density on nid's parent bond.
            ch = children[nid]
            arr = arrays[nid]
            q = qubit_of_node.get(nid)
            if q is not None:
                # p[B, x] = Re sum_{a,a'} rho[a,a'] T[a,x] conj(T[a',x]).
                flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
                p = self._einsum(
                    "BaA,axF,AxF->Bx", rho, flat, flat.conj()
                ).real
                p = self._clip_nonnegative(p)
                total = self._sum(p, axis=1, keepdims=True)
                safe_total = self._where(total > 0.0, total, 1.0)
                probs = p / safe_total
                draws = physical_draws[draw_index]
                draw_index += 1
                cdf = self._cumsum(probs, axis=1)
                x = self._sum(draws[:, None] > cdf, axis=1)
                x = self._as_int64(self._clip_max(x, probs.shape[1] - 1))
                configs[:, q] = x
                prob[:] *= probs[batch, x]
                # Selecting one physical value leaves a batched tensor over
                # the parent and child bonds. A physical leaf has no remaining
                # child axes and can return immediately.
                selected = self._moveaxis(arr[:, x, ...], 1, 0)
                if not ch:
                    return selected.reshape(B, arr.shape[0])

                par = arr.shape[0]
                K = selected
                start = 0
            else:
                if not ch:
                    raise ValueError(
                        f"virtual tree leaf {nid} has no physical qubit."
                    )

                par = arr.shape[0]
                # First child: trace the (unbatched) future siblings to
                # identity. This avoids broadcasting every virtual-only node
                # across the sample batch.
                d0 = arr.shape[1]
                F0 = int(np.prod(arr.shape[2:])) if len(ch) > 1 else 1
                ur = arr.reshape(par, d0, F0)
                env = self._einsum("acF,AdF->acAd", ur, ur.conj())
                rho0 = self._einsum("BaA,acAd->Bcd", rho, env)
                phi0 = visit(ch[0], rho0)
                # Collapse child 0 into the node tensor -> batched remainder.
                K = self._tensordot(phi0, arr, axes=([1], [1]))
                start = 1

            for i in range(start, len(ch)):
                di = K.shape[2]
                Fi = int(np.prod(K.shape[3:])) if K.ndim > 3 else 1
                Kf = K.reshape(B, par, di, Fi)
                X = self._einsum("BaA,BacF->BAcF", rho, Kf)
                rho_i = self._einsum("BAcF,BAdF->Bcd", X, Kf.conj())
                phi_i = visit(ch[i], rho_i)
                Knew = self._einsum("BpcF,Bc->BpF", Kf, phi_i)
                K = Knew.reshape((B, par) + K.shape[3:])
            return K.reshape(B, par)

        rho_root = self._ones((B, 1, 1), dtype=arrays[self._root].dtype)
        visit(self._root, rho_root)
        return configs, prob

    def sample_arrays(self, n_samples: int = 1, seed=None, *, to_numpy=False):
        """Draw samples and return raw ``(configs, probs)`` arrays.

        ``configs`` has shape ``(n_samples, nqubits)`` and ``probs`` has shape
        ``(n_samples,)``. By default arrays use the resolved sampler backend;
        pass ``to_numpy=True`` for an explicit host copy.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = self._rng if seed is None else np.random.default_rng(seed)
        with self._thread_ctx():
            if self.resolved_backend == "symmray":
                configs, probs = self._symmray_sample_arrays(int(n_samples), rng)
                if not to_numpy:
                    if self._symmray_state["array_backend"] == "torch":
                        import torch

                        configs = torch.as_tensor(
                            configs,
                            dtype=torch.int64,
                            device=self._symmray_state["template"].device,
                        )
                        probs = torch.as_tensor(
                            probs,
                            dtype=self._symmray_state["template"].real.dtype,
                            device=self._symmray_state["template"].device,
                        )
                    elif self._symmray_state["array_backend"] == "cupy":
                        import cupy as cp

                        configs = cp.asarray(configs, dtype=cp.int64)
                        probs = cp.asarray(
                            probs,
                            dtype=self._symmray_state["template"].real.dtype,
                        )
            else:
                configs, probs = self._sample_arrays(int(n_samples), rng)
        if to_numpy:
            return _backend_array_to_numpy(configs), _backend_array_to_numpy(probs)
        return configs, probs

    def sample_batch(
        self, n_samples: int = 1, seed=None, *, to_numpy=False
    ) -> TreeBatchSampleResult:
        """Draw samples and return a batched :class:`TreeBatchSampleResult`.

        This is the preferred API for fast downstream workflows; use
        :meth:`sample_arrays` when tuple unpacking is more convenient, or
        :meth:`sample` for the list-based result.
        """
        configs, probs = self.sample_arrays(
            n_samples, seed=seed, to_numpy=to_numpy
        )
        return TreeBatchSampleResult(
            configs=configs,
            probs=probs,
            nqubits=self._nqubits,
            configuration_encoding=self._configuration_encoding,
            backend=(
                "numpy"
                if to_numpy
                else (
                    self._symmray_state["array_backend"]
                    if self.resolved_backend == "symmray"
                    else self.resolved_backend
                )
            ),
        )

    def sample(self, n_samples: int = 1, seed=None) -> TreeSampleResult:
        """Draw ``n_samples`` configurations from the tree state.

        Returns
        -------
        TreeSampleResult
            Contains per-sample configs and exact Born probabilities.
        """
        return self.sample_batch(n_samples, seed=seed, to_numpy=True).to_sample_result()

    # -- evaluation ----------------------------------------------------------

    def _check_configs(self, configs):
        if self.resolved_backend == "symmray":
            configs = np.asarray(_backend_array_to_numpy(configs), dtype=np.int64)
        elif self.resolved_backend == "torch":
            import torch

            configs = torch.as_tensor(
                configs, dtype=torch.int64, device=self._device
            )
        elif self.resolved_backend == "cupy":
            import cupy as cp

            configs = cp.asarray(configs, dtype=cp.int64)
        else:
            configs = np.asarray(configs, dtype=np.int64)
        if configs.ndim != 2 or configs.shape[1] != self._nqubits:
            raise ValueError(
                f"configs must have shape (batch, nqubits={self._nqubits}); "
                f"got {tuple(configs.shape)}."
            )
        return configs

    def _amplitudes(self, configs):
        arrays = self._arrays
        children = self._children
        qubit_of_node = self._qubit_of_node
        B = configs.shape[0]

        def visit(nid):
            ch = children[nid]
            arr = arrays[nid]
            q = qubit_of_node.get(nid)
            if q is not None:
                x = configs[:, q]
                K = self._moveaxis(arr[:, x, ...], 1, 0)
                if not ch:
                    return K.reshape(B, arr.shape[0])
                start = 0
            else:
                if not ch:
                    raise ValueError(
                        f"virtual tree leaf {nid} has no physical qubit."
                    )
                K = None
                start = 0

            par = arr.shape[0]
            for i, child in enumerate(ch):
                phi_c = visit(child)
                if K is None and i == start:
                    K = self._tensordot(phi_c, arr, axes=([1], [1]))
                else:
                    di = K.shape[2]
                    Fi = int(np.prod(K.shape[3:])) if K.ndim > 3 else 1
                    Kf = K.reshape(B, par, di, Fi)
                    Knew = self._einsum("BpcF,Bc->BpF", Kf, phi_c)
                    K = Knew.reshape((B, par) + K.shape[3:])
            return K.reshape(B, par)

        return visit(self._root)[:, 0]

    def amplitudes(self, configs, *, to_numpy: bool = True):
        """Return amplitudes ``<config|psi>`` for batched ``configs``.

        ``configs`` should have shape ``(batch, nqubits)``.  The tree is
        contracted from the leaves to the root in one batched pass.  For a
        fermionic tree the codes index Symmray's dense basis order and the
        returned amplitude may differ from the graded amplitude ordering by a
        per-configuration sign; the derived :meth:`probabilities` are exact.
        """
        configs = self._check_configs(configs)
        with self._thread_ctx():
            out = (
                self._symmray_amplitudes(configs)
                if self.resolved_backend == "symmray"
                else self._amplitudes(configs)
            )
        return _backend_array_to_numpy(out) if to_numpy else out

    def probabilities(self, configs, *, to_numpy: bool = True):
        """Return Born probabilities ``|<config|psi>|**2`` for ``configs``.

        For the normalized state captured by the sampler this is the exact
        probability of each supplied configuration.
        """
        configs = self._check_configs(configs)
        with self._thread_ctx():
            if self.resolved_backend == "symmray":
                out = self._symmray_probabilities(configs)
            else:
                amps = self._amplitudes(configs)
                out = self._xp().abs(amps) ** 2
        return _backend_array_to_numpy(out) if to_numpy else out
