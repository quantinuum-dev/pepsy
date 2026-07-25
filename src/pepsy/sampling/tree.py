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

Fermionic states
----------------
Native Symmray fermionic trees are supported through the same ``O(L)`` sweep.
The graded-canonical tensors are densified once; because every Born probability
contracts a tensor with its own conjugate over the shared indices, the fermionic
exchange signs enter squared and cancel, so the plain dense sweep reproduces the
exact graded probabilities and marginals (validated to machine precision against
the doubled-network contraction).  Sampled physical codes follow Symmray's dense
basis order -- ``empty, up, down, up-down`` for spinful ``phys_dim=4`` and
``empty, occupied`` for spinless ``phys_dim=2`` -- and decode to ``(n_up,
n_down)`` occupations through the :class:`FermionConfigurationEncoding` attached
to the sample results.  Signed :meth:`~TreeSampler.amplitudes` use the same dense
basis convention and may differ from the graded amplitude ordering by a
per-configuration sign, whereas :meth:`~TreeSampler.probabilities` are exact.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from .samplers import FermionConfigurationEncoding

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
        Number of qubit leaves in the tree.
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
        Number of qubit leaves in the tree.
    """

    configs: np.ndarray
    probs: np.ndarray
    nqubits: int
    configuration_encoding: FermionConfigurationEncoding | None = None

    def __len__(self):
        return int(self.configs.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of sampled configurations."""
        return len(self)

    def to_numpy(self) -> "TreeBatchSampleResult":
        """Return a plain NumPy copy of this batched result."""
        return TreeBatchSampleResult(
            configs=np.asarray(self.configs),
            probs=np.asarray(self.probs),
            nqubits=self.nqubits,
            configuration_encoding=self.configuration_encoding,
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
        return [[int(value) for value in config] for config in self.configs]

    def magnetizations(self) -> np.ndarray:
        """Per-sample magnetization ``(1 / n) * sum_i (1 - 2 * bit_i)``."""
        configs = np.asarray(self.configs, dtype=float)
        return (1 - 2 * configs).sum(axis=1) / float(self.nqubits)

    def to_sample_result(self) -> TreeSampleResult:
        """Convert to the list-based :class:`TreeSampleResult`."""
        return TreeSampleResult(
            configs=self.configs_list(),
            probs=[float(p) for p in np.asarray(self.probs)],
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
    fermion : pepsy.tensors.Fermion, optional
        Fermionic physical-space convention.  It is optional: a native Symmray
        fermionic tree is detected automatically and its dense-basis occupation
        decoder is attached to the sample results regardless.  Supplying a
        ``fermion`` only pins the recorded ``symmetry``/``spinful`` labels when
        they cannot be inferred from the state.

    Notes
    -----
    The node arrays are extracted once (in canonical, normalized form) and
    cached.  Call :meth:`refresh` after the source state changes; otherwise the
    sampler keeps representing its previously captured tensor data.
    """

    def __init__(self, state, *, seed=None, threads: int | None = 1, fermion=None):
        self._rng = np.random.default_rng(seed)
        self.threads = None if threads is None else int(threads)
        self._fermion = fermion

        # Geometry / cached canonical arrays (populated by refresh).
        self._nqubits = None
        self._root = None
        self._children = None
        self._qubit_of_leaf = None
        self._arrays = None
        self._fermionic = False
        self._configuration_encoding = None

        self.refresh(state)

    # -- setup ---------------------------------------------------------------

    def _thread_ctx(self):
        """Context manager capping BLAS/OpenMP threads for the batched ops."""
        if _THREAD_CONTROLLER is not None and self.threads is not None:
            return _THREAD_CONTROLLER.limit(limits=self.threads)
        return contextlib.nullcontext()

    @staticmethod
    def _resolve_ttn(state):
        """Return a canonical, normalized :class:`TreeTensorNetwork` copy."""
        tn = getattr(state, "tn", None)
        if tn is None:
            tn = state
        if not hasattr(tn, "plan") or not hasattr(tn, "node_tensor"):
            raise TypeError(
                "TreeSampler expects a TreeTensorNetwork or a TreeOptimizer; "
                f"got {type(state).__name__}."
            )
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
            tn = self._resolve_ttn(state)
            self._extract_arrays(tn)
        return self

    def _extract_arrays(self, tn):
        """Extract per-node dense arrays with a canonical axis order.

        Leaf arrays have axes ``(parent_bond, phys)``; internal arrays have
        axes ``(parent_bond, child_0, child_1, ...)``.  The root is given a
        dummy parent bond of size 1 so the sampling recursion is uniform.  The
        root array is L2-normalized in place, which normalizes the whole state
        because every other tensor is isometric in this canonical form.

        Bond indices are resolved by intersecting adjacent node tensors rather
        than by the deterministic ``_tb{lo}_{hi}`` names, because gate threading
        and canonicalisation replace those with fresh ``quimb`` uuids.
        """
        plan = tn.plan
        root = plan.root
        children = {nid: tuple(plan.children[nid]) for nid in plan.children}
        qubit_of_leaf = dict(plan.qubit_of_leaf)

        # Native Symmray fermionic trees keep block-sparse arrays; densify them
        # once.  A graded-canonical tensor is also plain-isometric (its exchange
        # signs cancel in the ket-with-bra reduced-density contraction), so the
        # dense sweep below reproduces the exact fermionic Born probabilities.
        fermionic = bool(getattr(tn, "fermionic", False))
        if fermionic:
            def to_arr(data):
                return np.asarray(data.to_dense())
        else:
            def to_arr(data):
                return np.asarray(data)

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
            if not ch:  # leaf
                phys = tn.site_ind(qubit_of_leaf[nid])
                if is_root:  # single-qubit tree
                    arr = to_arr(t.transpose(phys).data).reshape(1, -1)
                else:
                    pbond = bond_between(nid, parent)
                    arr = to_arr(t.transpose(pbond, phys).data)
            else:  # internal node
                cbonds = [bond_between(nid, c) for c in ch]
                if is_root:
                    arr = to_arr(t.transpose(*cbonds).data)
                    arr = arr.reshape((1,) + arr.shape)
                else:
                    pbond = bond_between(nid, parent)
                    arr = to_arr(t.transpose(pbond, *cbonds).data)
            arrays[nid] = arr

        # Normalize via the root array (state is canonical with centre = root).
        root_arr = arrays[root]
        nrm = float(np.sqrt(np.sum(np.abs(root_arr) ** 2)))
        if nrm > 0:
            arrays[root] = root_arr / nrm

        self._nqubits = int(plan.n)
        self._root = root
        self._children = children
        self._qubit_of_leaf = qubit_of_leaf
        self._arrays = arrays
        self._fermionic = fermionic
        self._configuration_encoding = (
            self._build_configuration_encoding(tn, arrays, qubit_of_leaf)
            if fermionic
            else None
        )

    def _build_configuration_encoding(self, tn, arrays, qubit_of_leaf):
        """Build the dense-basis occupation decoder for a fermionic tree."""
        if not qubit_of_leaf:
            return None
        phys_dim = int(arrays[next(iter(qubit_of_leaf))].shape[-1])
        fermion = self._fermion
        if fermion is not None and hasattr(fermion, "spinful"):
            spinful = bool(fermion.spinful)
        else:
            spinful = phys_dim == 4
        table = _fermion_code_occupations(phys_dim, spinful)
        if table is None:
            return None
        symmetry = getattr(fermion, "symmetry", None)
        if symmetry is None:
            symmetry = getattr(tn, "symmetry", None)
        symmetry = str(symmetry) if symmetry is not None else "U1U1"
        return FermionConfigurationEncoding(
            symmetry=symmetry,
            spinful=spinful,
            code_to_occupations=tuple(table for _ in range(self._nqubits)),
        )

    @property
    def nqubits(self) -> int:
        """Number of qubit leaves in the tree."""
        return self._nqubits

    # -- sampling ------------------------------------------------------------

    def _sample_arrays(self, n_samples, rng):
        """Batched perfect sampling; returns ``(configs, probs)`` arrays."""
        B = int(n_samples)
        arrays = self._arrays
        children = self._children
        qubit_of_leaf = self._qubit_of_leaf
        configs = np.zeros((B, self._nqubits), dtype=np.int64)
        prob = np.ones(B, dtype=np.float64)
        batch = np.arange(B)

        def visit(nid, rho):
            # rho: (B, d_par, d_par) reduced density on nid's parent bond.
            ch = children[nid]
            arr = arrays[nid]
            if not ch:  # leaf
                # p[B, x] = Re sum_{a,a'} rho[a,a'] T[a,x] conj(T[a',x]).
                tmp = np.einsum("BaA,ax->BAx", rho, arr)
                p = np.einsum("BAx,Ax->Bx", tmp, arr.conj()).real
                p = np.clip(p, 0.0, None)
                total = p.sum(axis=1, keepdims=True)
                probs = p / np.where(total > 0.0, total, 1.0)
                draws = rng.random(B)
                cdf = np.cumsum(probs, axis=1)
                x = (draws[:, None] > cdf).sum(axis=1)
                x = np.minimum(x, probs.shape[1] - 1).astype(np.int64)
                configs[:, qubit_of_leaf[nid]] = x
                prob[:] *= probs[batch, x]
                # phi[B, a] = T[a, x] -- the collapsed subtree amplitude.
                return arr[:, x].T

            par = arr.shape[0]
            # First child: trace the (unbatched) future siblings to identity.
            d0 = arr.shape[1]
            F0 = int(np.prod(arr.shape[2:])) if len(ch) > 1 else 1
            ur = arr.reshape(par, d0, F0)
            env = np.einsum("acF,AdF->acAd", ur, ur.conj())
            rho0 = np.einsum("BaA,acAd->Bcd", rho, env)
            phi0 = visit(ch[0], rho0)
            # Collapse child 0 into the node tensor -> batched remainder.
            K = np.tensordot(phi0, arr, axes=([1], [1]))  # (B, par, d1, ...)
            for i in range(1, len(ch)):
                di = K.shape[2]
                Fi = int(np.prod(K.shape[3:])) if K.ndim > 3 else 1
                Kf = K.reshape(B, par, di, Fi)
                X = np.einsum("BaA,BacF->BAcF", rho, Kf)
                rho_i = np.einsum("BAcF,BAdF->Bcd", X, Kf.conj())
                phi_i = visit(ch[i], rho_i)
                Knew = np.einsum("BpcF,Bc->BpF", Kf, phi_i)
                K = Knew.reshape((B, par) + K.shape[3:])
            return K.reshape(B, par)

        rho_root = np.ones((B, 1, 1), dtype=complex)
        visit(self._root, rho_root)
        return configs, prob

    def sample_arrays(self, n_samples: int = 1, seed=None):
        """Draw samples and return raw ``(configs, probs)`` NumPy arrays.

        ``configs`` has shape ``(n_samples, nqubits)`` and ``probs`` has shape
        ``(n_samples,)``.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = self._rng if seed is None else np.random.default_rng(seed)
        with self._thread_ctx():
            return self._sample_arrays(int(n_samples), rng)

    def sample_batch(
        self, n_samples: int = 1, seed=None
    ) -> TreeBatchSampleResult:
        """Draw samples and return a batched :class:`TreeBatchSampleResult`.

        This is the preferred API for fast downstream workflows; use
        :meth:`sample_arrays` when tuple unpacking is more convenient, or
        :meth:`sample` for the list-based result.
        """
        configs, probs = self.sample_arrays(n_samples, seed=seed)
        return TreeBatchSampleResult(
            configs=configs,
            probs=probs,
            nqubits=self._nqubits,
            configuration_encoding=self._configuration_encoding,
        )

    def sample(self, n_samples: int = 1, seed=None) -> TreeSampleResult:
        """Draw ``n_samples`` configurations from the tree state.

        Returns
        -------
        TreeSampleResult
            Contains per-sample configs and exact Born probabilities.
        """
        return self.sample_batch(n_samples, seed=seed).to_sample_result()

    # -- evaluation ----------------------------------------------------------

    def _check_configs(self, configs):
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
        qubit_of_leaf = self._qubit_of_leaf
        B = configs.shape[0]

        def visit(nid):
            ch = children[nid]
            arr = arrays[nid]
            if not ch:  # leaf
                x = configs[:, qubit_of_leaf[nid]]
                return arr[:, x].T  # (B, d_par)
            par = arr.shape[0]
            K = None
            for i, child in enumerate(ch):
                phi_c = visit(child)
                if i == 0:
                    K = np.tensordot(phi_c, arr, axes=([1], [1]))
                else:
                    di = K.shape[2]
                    Fi = int(np.prod(K.shape[3:])) if K.ndim > 3 else 1
                    Kf = K.reshape(B, par, di, Fi)
                    Knew = np.einsum("BpcF,Bc->BpF", Kf, phi_c)
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
            out = self._amplitudes(configs)
        return np.asarray(out) if to_numpy else out

    def probabilities(self, configs, *, to_numpy: bool = True):
        """Return Born probabilities ``|<config|psi>|**2`` for ``configs``.

        For the normalized state captured by the sampler this is the exact
        probability of each supplied configuration.
        """
        amps = self.amplitudes(configs, to_numpy=to_numpy)
        return np.abs(amps) ** 2
