"""MPO optimization helpers centered on :class:`MpoOptimizer`.

:class:`MpoOptimizer` replays a queue of gates ``[(gate, where), ...]`` against
an MPO ``O`` of length ``L`` with two physical index families (``ind_id_k``,
``ind_id_b``).  Each bundled entry specifies what acts on the ket and bra
legs:

* ``gate``                       → apply ``gate`` on ket and ``gate†`` on bra
  (default "unitary conjugation" semantics ``G O G†``);
* ``(gate,)`` or ``(gate, None)`` → apply ``gate`` on ket only;
* ``(None, B)``                  → apply ``B†`` on bra only;
* ``(G, B)``                     → apply ``G`` on ket and ``B†`` on bra.

Three execution backends are supported, all returning the same kind of MPO
but differing in *how* two-site updates are compressed back to bond ``chi``:

* ``mode="dmrg"`` — fit a target MPO with :class:`pepsy.fitting.local.FIT`
  inside a local window ``[xmin, xmax]``; supports batching consecutive
  two-site gates via ``k_2q_batch``;
* ``mode="svd"``  — apply the gate with ``reduce-split`` then canonicalize +
  left-compress to ``chi``;
* ``mode="mpo"``  — use :func:`pepsy.operators.gates.gate_nonlocal_opt` to
  apply each layer independently on the ket and bra families. Symmray MPOs
  use the block-aware SVD path instead.

The class also tracks a running "normalized-norm" proxy
``sqrt(<O|O> / <O0|O0>)`` that equals ``1`` for purely unitary two-sided
evolution (useful as a quick sanity signal).
"""

from __future__ import annotations

from numbers import Integral

import autoray as ar
import numpy as np

from ...tensors.core import tn_norm
from ...fitting.local import FIT
from ...operators.gates import _normalize_gate_entries, gate as apply_gate, gate_nonlocal_opt

__all__ = ["MpoOptimizer"]


def _normalize_gate_queue(gates):
    """Return ``(gate_list, where_list)`` from canonical bundled stream input."""
    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], []
    gate_list, where_list = zip(*entries)

    def normalize_where(where):
        if isinstance(where, Integral):
            return (int(where),)
        if isinstance(where, list):
            return tuple(where)
        return where

    return list(gate_list), [normalize_where(where) for where in where_list]


class MpoOptimizer:
    """High-level wrapper for MPO gate sweeps.

    Parameters
    ----------
    mpo : qtn.MatrixProductOperator
        Initial MPO ``O``.  Used as the starting point for the queued
        evolution.  By default a copy is taken (see ``inplace``).
    gates : sequence | int | None, optional
        Canonical bundled gate stream ``[(gate, where), ...]``.  ``gate``
        encodes the (ket, bra) action per entry, with each side optionally
        ``None``:

        * ``G``            → apply ``G`` on ket and ``G†`` on bra (``G O G†``);
        * ``(G,)``         → ket-only shorthand for ``(G, None)``;
        * ``(G, None)``    → apply ``G`` on ket only;
        * ``(None, B)``    → apply ``B†`` on bra only;
        * ``(G, B)``       → apply ``G`` on ket and ``B†`` on bra.

        For backward compatibility, passing a bare ``int`` is treated as
        ``chi`` with an empty gate queue.
    chi : int
        Working bond dimension used by all compression backends.
    mode : {"dmrg", "svd", "mpo"}, default="dmrg"
        Execution backend for two-site updates (see module docstring).
    ind_id_k : str, default="k{}"
        Site-index format string for the ket physical leg family.
    ind_id_b : str, default="b{}"
        Site-index format string for the bra physical leg family.
    contraction_opt : object | None, optional
        Contraction path optimizer keyword used by :func:`tn_norm` and
        :class:`pepsy.fitting.local.FIT`.  Defaults to ``"auto-hq"``.
    inplace : bool, default=False
        When ``True`` mutate ``mpo`` directly; otherwise operate on a copy
        and leave the input untouched.

    Attributes
    ----------
    p : qtn.MatrixProductOperator
        Current MPO state (after construction and after each :meth:`run`).
    G, where : list
        Parsed gate-tensor list and corresponding site-coordinate list.
    losses : list[float]
        Running history of the normalized-norm proxy
        ``sqrt(<O|O> / <O0|O0>)`` appended at sampled steps during a run.
    info_c : dict
        Cached canonicalization metadata (``cur_orthog`` tracks the current
        orthogonality center / span).
    """

    _ALLOWED_MODES = frozenset({"dmrg", "svd", "mpo"})

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` is a Symmray block-sparse array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @staticmethod
    def _is_fermionic_array(value):
        """Return whether ``value`` carries Symmray graded-array metadata."""
        return bool(getattr(value, "fermionic", False)) or (
            "fermionicarray" in type(value).__name__.lower()
        )

    @classmethod
    def _has_symmray_data(cls, tn):
        """Return whether any tensor in ``tn`` stores Symmray data."""
        return any(
            cls._is_symmray_array(getattr(tensor, "data", None))
            for tensor in getattr(tn, "tensors", ())
        )

    @staticmethod
    def _resolve_cutoff(value, p):
        """Resolve a numeric or dtype-aware truncation cutoff."""
        if value == "auto":
            dtype_names = [
                str(getattr(tensor.data, "dtype", "")).lower()
                for tensor in p.tensors
            ]
            if any("16" in dtype for dtype in dtype_names):
                return 1.0e-3
            if any("32" in dtype or "complex64" in dtype for dtype in dtype_names):
                return 1.0e-6
            return 1.0e-12
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cutoff must be 'auto' or a non-negative number."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("cutoff must be 'auto' or a non-negative number.")
        return value

    @staticmethod
    def _resolve_fit_max_span(value, k_2q_batch):
        """Resolve the maximum inclusive spatial span of a DMRG batch."""
        if value is None:
            return None
        if value == "auto":
            return max(3, 2 * int(k_2q_batch) + 1)
        if not isinstance(value, Integral) or int(value) < 2:
            raise ValueError(
                "fit_max_span must be 'auto', None, or an integer >= 2."
            )
        return int(value)

    @classmethod
    def _normalize_mode(cls, mode):
        """Lower-case and validate ``mode`` against :attr:`_ALLOWED_MODES`."""
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            supported = ", ".join(sorted(cls._ALLOWED_MODES))
            raise ValueError(f"Unknown mode: {mode}. Supported modes: {supported}")
        return mode_norm

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mpo,
        gates=None,
        chi=None,
        mode="dmrg",
        ind_id_k="k{}",
        ind_id_b="b{}",
        contraction_opt=None,
        inplace=False,
    ):
        # Allow the shorthand ``MpoOptimizer(mpo, chi)``: bare int second arg
        # is interpreted as chi with an empty gate queue.
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpoOptimizer(mpo, gates, chi) "
                    "or MpoOptimizer(mpo, chi) for an empty gate queue."
                )

        self.inplace = bool(inplace)
        # Work on a copy by default so the user's input MPO stays unchanged.
        self.p = mpo if self.inplace else mpo.copy()
        self.G, self.where = _normalize_gate_queue(gates)
        self.chi = int(chi)
        self.mode = self._normalize_mode(mode)
        self.ind_id_k = str(ind_id_k)
        self.ind_id_b = str(ind_id_b)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt

        # Reference norm used to normalize the loss proxy in `_normalize_norm`.
        self.norm_mpo = self._measure_norm(self.p)
        self.info_c = {}
        self.losses = [1.0]
        self._init_canonicalization()

    def _current_orthog(self, p=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Accepts cached entries shaped as ``"calc"`` / ``None`` (recompute),
        ``int`` (single site), or 1- and 2-tuples.  The canonical form
        returned and stored back into ``self.info_c['cur_orthog']`` is always
        a 2-tuple with ``min <= max``.
        """
        cur = self.info_c.get("cur_orthog", "calc")
        state = self.p if p is None else p
        if cur == "calc" or cur is None:
            lo, hi = state.calc_current_orthog_center()
            cur = (int(lo), int(hi))
        elif isinstance(cur, Integral):
            cur = (int(cur), int(cur))
        elif len(cur) == 1:
            cur = (int(cur[0]), int(cur[0]))
        elif len(cur) == 2:
            cur = (int(min(cur)), int(max(cur)))
        else:
            raise ValueError("cur_orthog must be an int, (int,), or (int, int).")

        self.info_c["cur_orthog"] = cur
        return cur

    def _init_canonicalization(self):
        """Put ``self.p`` into mixed-canonical form with center at ``L // 2``."""
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def _prepare_dmrg_state(self, fit_block_size=1):
        """Prepare the MPO for the selected local FIT update.

        The historical dense one-site FIT path has fixed bond dimensions and
        still needs the old eager expansion. Native Symmray paths cannot use
        that dense padding, and native two- and three-site FIT splits discover
        the required bond sectors themselves, so native paths never pre-expand
        their multi-sector bonds.
        """
        if int(fit_block_size) != 1 or self._has_symmray_data(self.p):
            return
        if self.p.max_bond() < self.chi:
            self.p.expand_bond_dimension(self.chi, inplace=True)
            self._init_canonicalization()

    def set_mpo(self, mpo):
        """Assign a new MPO and reset canonicalization metadata."""
        self.p = mpo if self.inplace else mpo.copy()
        self.norm_mpo = self._measure_norm(self.p)
        self._init_canonicalization()
        return self

    def set_mode(self, mode):
        """Set execution mode."""
        self.mode = self._normalize_mode(mode)
        return self

    def set_gates(self, gates):
        """Replace the current gate queue with canonical bundled entries."""
        self.G, self.where = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append canonical bundled entries to the existing gate queue."""
        G_new, where_new = _normalize_gate_queue(gates)
        self.G.extend(G_new)
        self.where.extend(where_new)
        return self

    @staticmethod
    def _normalize_fidelity_samples(fidelity_samples):
        """Validate and normalize fidelity-sample count."""
        samples = int(fidelity_samples)
        if samples < 0:
            raise ValueError("fidelity_samples must be >= 0.")
        return samples

    @staticmethod
    def _sampling_steps(total_steps, fidelity_samples):
        """Return the set of gate-step indices at which to sample the norm proxy.

        Indices are 1-based gate counts (``1 ≤ step ≤ total_steps``).  The
        final step is always included so the run history ends with a fresh
        measurement; up to ``fidelity_samples`` extra interior points are
        spread linearly across the remaining range.
        """
        if total_steps <= 0:
            return set()

        samples = MpoOptimizer._normalize_fidelity_samples(fidelity_samples)
        sample_steps = set()

        if total_steps > 1 and samples > 0:
            interior_count = min(samples, total_steps - 1)
            for step in np.linspace(1, total_steps - 1, num=interior_count, dtype=int):
                sample_steps.add(int(step))

        sample_steps.add(total_steps)
        return sample_steps

    @staticmethod
    def _real_float(value):
        """Convert backend scalar/tensor-like values to Python float (real part)."""
        real_value = ar.do("real", value)
        item = getattr(real_value, "item", None)
        if callable(item):
            try:
                real_value = item()
            except TypeError:
                pass
        return float(real_value)

    def _append_norm_proxy_sample(self, p):
        """Append current normalized MPO norm and return it."""
        norm_val = self._normalize_norm(self._measure_norm(p))
        self.losses.append(norm_val)
        return norm_val

    def _measure_norm(self, p):
        """Return ``(mantissa, exponent)`` such that ``<O|O> = mantissa * 10**exponent``.

        Working in log-mantissa form keeps the proxy stable for very long
        gate streams where ``<O|O>`` can over- or under-flow.
        """
        mantissa, exponent = tn_norm(p, contraction_opt=self.contraction_opt, strip_exponent=True)
        return self._real_float(mantissa), float(exponent)

    def _normalize_norm(self, norm_val):
        """Convert a ``(mantissa, exponent)`` squared-norm into the relative MPO norm.

        Returns ``sqrt(<O|O>) / sqrt(<O0|O0>)`` — i.e. the ratio of actual
        MPO norms relative to the construction-time reference ``norm_mpo``.
        For purely unitary two-sided evolution this stays equal to ``1``.
        """
        m, e = norm_val
        m0, e0 = self.norm_mpo
        if m0 == 0.0:
            return 0.0 if m == 0.0 else float("inf")
        return float(np.sqrt(abs(m / m0))) * 10 ** ((e - e0) / 2)

    @staticmethod
    def _prepare_gate_tensor(gate, n_sites):
        """Reorder a gate tensor into ``(input, output)`` ket-index order.

        Quimb gates are stored as ``(output, input)`` matrices (or rank-4
        ``(o1, o2, i1, i2)`` tensors for two-site gates).  ``apply_gate``
        below expects the opposite ordering, so we transpose accordingly.
        """
        # Native Symmray gates already carry explicit dual metadata describing
        # output and input legs. Transposing them as if they were dense Quimb
        # gates changes the charge sectors (and can make a U1U1 split ask for
        # impossible virtual charges). Keep their graded/block structure intact.
        if MpoOptimizer._is_symmray_array(gate):
            return gate

        if n_sites == 1:
            return ar.do("transpose", gate, (1, 0))
        elif n_sites == 2:
            shape = getattr(gate, "shape", ())
            if len(shape) == 2:
                # 4x4 matrix form: just a matrix transpose.
                din, dout = shape
                if int(din) != int(dout):
                    raise ValueError(
                        "Two-site gate matrix must be square with shape (d**2, d**2)."
                    )
                return ar.do("transpose", gate, (1, 0))
            elif len(shape) == 4:
                # Rank-4 form: swap output and input pairs.
                return ar.do("transpose", gate, (2, 3, 0, 1))
            else:
                raise ValueError(
                    "Two-site gate must have shape (d**2, d**2) or (d, d, d, d)."
                )
        else:
            raise ValueError("Each gate location must have one or two sites.")

    @classmethod
    def _symmray_physical_map(cls, p, site, ind_id):
        """Return the dense charge list for one live MPO physical index."""
        ind = ind_id.format(site)
        tensor = next(
            tensor
            for tensor in getattr(p, "tensors", ())
            if ind in getattr(tensor, "inds", ())
        )
        axis = tensor.inds.index(ind)
        chargemap = tensor.data.indices[axis].chargemap
        return [charge for charge, size in chargemap.items() for _ in range(int(size))]

    @staticmethod
    def _charge_parity(charge):
        """Return fermion parity inferred from a scalar or product charge."""
        if isinstance(charge, tuple):
            return sum(int(part) for part in charge) % 2
        return int(charge) % 2

    @classmethod
    def _fermionic_gate_to_bosonic(cls, p, gate, where, ind_id):
        """Convert an even native gate for a non-graded Symmray MPO.

        ``SymHamiltonian.to_mpo`` deliberately returns a bosonic Symmray MPO
        with the Jordan-Wigner parity convention already encoded in its local
        channels. Native ``Fermion`` gates use Symmray's graded tensor-product
        convention instead. For an even two-site gate, changing conventions
        amounts to the endpoint crossing phase on the input ket sectors.
        """
        if not cls._is_fermionic_array(gate) or not cls._has_symmray_data(p):
            return gate

        sample = next(
            tensor.data
            for tensor in getattr(p, "tensors", ())
            if cls._is_symmray_array(getattr(tensor, "data", None))
        )
        if cls._is_fermionic_array(sample):
            return gate

        try:
            dense = gate.to_dense()
        except AttributeError:
            dense = gate
        dense = np.asarray(ar.to_numpy(dense))
        where = tuple(where)
        physical_maps = [
            cls._symmray_physical_map(p, site, ind_id) for site in where
        ]
        n_sites = len(where)
        if n_sites == 2:
            left_odd = np.array(
                [cls._charge_parity(charge) for charge in physical_maps[0]],
                dtype=bool,
            )
            right_odd = np.array(
                [cls._charge_parity(charge) for charge in physical_maps[1]],
                dtype=bool,
            )
            crossing = np.ones(
                (len(left_odd), len(right_odd)),
                dtype=dense.dtype,
            )
            crossing[np.ix_(left_odd, right_odd)] = -1
            dense = dense * crossing[None, None, :, :]

        import symmray.utils as sr_utils  # pylint: disable=import-outside-toplevel

        sample_charge = next(iter(physical_maps[0]), 0)
        zero = (
            tuple(0 for _ in sample_charge)
            if isinstance(sample_charge, tuple)
            else 0
        )
        return sr_utils.from_dense(
            dense,
            symmetry=getattr(sample, "symmetry", None),
            index_maps=physical_maps * 2,
            duals=(False,) * n_sites + (True,) * n_sites,
            fermionic=False,
            charge=zero,
        )

    @classmethod
    def _prepare_gate_pair(cls, gate, n_sites, bra_gate=None, *, p=None, where=None,
                           ind_id="k{}"):
        """Return ``(g_k, g_b)`` ready to be fed to :func:`apply_gate`.

        ``gate`` becomes ``g_k`` (acts on the ket index family with the
        ket-ordering convention) and ``bra_gate`` becomes
        ``g_b = conj(prepare(bra_gate))``, which when applied to the bra
        family realises ``B† O`` on that side.  Passing ``None`` skips that
        side; at least one of the two must be provided.
        """
        if gate is None and bra_gate is None:
            raise ValueError("At least one of ket gate or bra gate must be provided.")

        if p is not None and where is not None:
            gate = (
                None
                if gate is None
                else cls._fermionic_gate_to_bosonic(p, gate, where, ind_id)
            )
            bra_gate = (
                None
                if bra_gate is None
                else cls._fermionic_gate_to_bosonic(p, bra_gate, where, ind_id)
            )

        g_k = None if gate is None else cls._prepare_gate_tensor(gate, n_sites)
        if bra_gate is None:
            g_b = None
        else:
            g_b = ar.do("conj", cls._prepare_gate_tensor(bra_gate, n_sites))
        return g_k, g_b

    @classmethod
    def _materialize_split_gate(cls, p, where):
        """Contract lazy native split-gate tensors back into site tensors.

        Quimb's ``split-gate`` application is deliberately lazy: for a
        non-local gate it temporarily leaves an extra tensor at each endpoint
        carrying the site tag. That is useful for general tensor-network
        workflows, but ``MatrixProductOperator.right_canonize_site`` expects
        exactly one tensor per site. Native Symmray arrays cannot use the
        dense swap-gate fallback, so materialize the endpoint tensors before
        the MPO canonicalization/compression sweep.
        """
        if not cls._has_symmray_data(p):
            return p

        for site in set(where):
            tag = f"I{site}"
            if len(p.tag_map.get(tag, ())) > 1:
                p.contract_tags(
                    tag,
                    preserve_tensor=True,
                    inplace=True,
                )
        return p

    @staticmethod
    def _parse_gate_entry(G_i, where_i):
        """Decompose one stream entry into ``(ket_gate, bra_gate, where)``.

        Acceptable shapes for ``G_i`` mirror the constructor convention:

        * bare ``G``        → ``(G, G)``  ("unitary conjugation" default);
        * ``(G,)``          → ``(G, None)`` (ket-only);
        * ``(G, B)``        → explicit pair, either side may be ``None``.

        At least one of the two sides must be non-``None``.
        """
        where_norm = tuple(where_i)
        if len(where_norm) not in (1, 2):
            raise ValueError("Each gate location must have one or two sites.")

        if isinstance(G_i, (tuple, list)):
            if len(G_i) == 1:
                # Explicit ket-only shorthand: (G,) -> (G, None)
                gate, bra_gate = G_i[0], None
            elif len(G_i) == 2:
                gate, bra_gate = G_i
            else:
                raise ValueError("Each MPO gate entry must be G, (G,), or (G, B).")
        else:
            # Default MPO evolution applies U on ket and U† on bra.
            gate, bra_gate = G_i, G_i

        if gate is None and bra_gate is None:
            raise ValueError("Each gate entry must provide at least one of G or B.")
        return gate, bra_gate, where_norm

    def _apply_gate_pair(
        self,
        p,
        gate,
        where,
        bra_gate=None,
        *,
        cutoff,
        cutoff_mode="rsum2",
        contract,
        inplace=True,
    ):
        """Apply the (ket, bra) gate pair onto ``p`` using :func:`apply_gate`.

        Each side is applied independently with its own ``ind_id_*`` so the
        two index families stay decoupled.
        """
        n_sites = len(where)
        g_k, g_b = self._prepare_gate_pair(
            gate,
            n_sites,
            bra_gate=bra_gate,
            p=p,
            where=where,
            ind_id=self.ind_id_k,
        )

        if g_k is not None:
            apply_gate(
                p,
                g_k,
                where,
                ind_id=self.ind_id_k,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )
        if g_b is not None:
            apply_gate(
                p,
                g_b,
                where,
                ind_id=self.ind_id_b,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )

        if contract == "split-gate":
            self._materialize_split_gate(p, where)

    def _build_dmrg_target(
        self,
        p,
        gate,
        where,
        bra_gate,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_cutoff=None,
    ):
        """Return ``p`` with one two-site gate pair applied via ``split-gate``.

        The result is the *target* MPO that the local FIT update will fit
        back to bond dimension ``chi`` inside the gate window.
        """
        if target_cutoff is None:
            target_cutoff = cutoff
        p_g = p.copy()
        self._apply_gate_pair(
            p_g,
            gate,
            where,
            bra_gate=bra_gate,
            cutoff=target_cutoff,
            cutoff_mode=cutoff_mode,
            contract="split-gate",
            inplace=True,
        )
        return p_g

    @staticmethod
    def _collect_dmrg_batch(
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        max_span=None,
    ):
        """Greedily collect up to ``k_2q_batch`` consecutive two-site gates.

        Any one-site gates encountered along the way are folded into the
        same batch so they are applied together inside a single FIT window.
        Returns ``(batch_G, batch_where, n_two_qubit_in_batch, next_idx)``.
        """
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = tuple(where_seq[idx])
            gate = G_seq[idx]
            if max_span is not None and batch_where:
                sites = [site for previous in batch_where for site in previous]
                sites.extend(where)
                proposed_span = max(sites) - min(sites) + 1
                if proposed_span > int(max_span):
                    break
            if len(where) == 1:
                batch_where.append(where)
                batch_G.append(gate)
            elif len(where) == 2:
                batch_where.append(where)
                batch_G.append(gate)
                two_qubit_in_batch += 1
            else:
                raise ValueError("Each gate location must have one or two sites.")
            idx += 1

        return batch_G, batch_where, two_qubit_in_batch, idx

    def _build_dmrg_batch_target(
        self,
        p,
        batch_G,
        batch_where,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_cutoff=None,
    ):
        """Apply a collected DMRG batch onto a copy of ``p``.

        Used to materialise the local target MPO for a batched FIT update.
        Two-site gates are split (``split-gate``) and one-site gates are
        contracted directly.
        """
        if target_cutoff is None:
            target_cutoff = cutoff
        p_g = p.copy()
        for G_i, where_i in zip(batch_G, batch_where):
            gate, bra_gate, where = self._parse_gate_entry(G_i, where_i)
            contract = True if len(where) == 1 else "split-gate"
            self._apply_gate_pair(
                p_g,
                gate,
                where,
                bra_gate=bra_gate,
                cutoff=target_cutoff,
                cutoff_mode=cutoff_mode,
                contract=contract,
                inplace=True,
            )
        return p_g

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        fidelity_samples=10,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        fit_max_span=None,
        fit_three_site_sweeps=1,
        target_cutoff=0.0,
    ):
        """Sweep the gate stream with local DMRG-style FIT compression.

        One-site gates are applied exactly; each two-site gate (or batch of
        ``k_2q_batch`` consecutive ones) is fitted by :class:`FIT` back to
        bond ``chi`` inside the gate window ``[xmin, xmax]``. ``fit_block_size``
        selects the one-, two-, or three-site native SVD update, and its
        cutoff policy is forwarded explicitly to both SVD splits and the
        requested sweep sequence.
        """
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")
        if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {
            1,
            2,
            3,
        }:
            raise ValueError("fit_block_size must be 1, 2, or 3.")
        fit_block_size = int(fit_block_size)

        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="dmrg_mpo",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
                idx += 1
                advanced = 1
            elif n_sites == 2:
                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    self.canonize_mpo(p, (xmin, xmax))
                    p_g = self._build_dmrg_target(
                        p,
                        gate,
                        where,
                        bra_gate,
                        cutoff,
                        cutoff_mode,
                        target_cutoff=target_cutoff,
                    )

                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                    )
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    fit.run_gate(
                        n_iter=n_iter,
                        verbose=False,
                        block_size=active_fit_block_size,
                        sweep_sequence=fit_sweep_sequence,
                        max_bond=self.chi,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        three_site_sweeps=fit_three_site_sweeps,
                    )
                    p = fit.p

                    self.info_c["cur_orthog"] = (xmin, xmax)
                    idx += 1
                    advanced = 1
                else:
                    batch_G, batch_where, two_qubit_in_batch, next_idx = (
                        self._collect_dmrg_batch(
                            G_seq,
                            where_seq,
                            idx,
                            k_2q_batch,
                            max_span=fit_max_span,
                        )
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for where_i in batch_where for site in where_i]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    self.canonize_mpo(p, (xmin, xmax))
                    p_g = self._build_dmrg_batch_target(
                        p,
                        batch_G,
                        batch_where,
                        cutoff,
                        cutoff_mode,
                        target_cutoff=target_cutoff,
                    )

                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                    )
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    fit.run_gate(
                        n_iter=n_iter,
                        verbose=False,
                        block_size=active_fit_block_size,
                        sweep_sequence=fit_sweep_sequence,
                        max_bond=self.chi,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        three_site_sweeps=fit_three_site_sweeps,
                    )
                    p = fit.p

                    self.info_c["cur_orthog"] = (xmin, xmax)
                    advanced = next_idx - idx
                    idx = next_idx
            else:
                raise ValueError("Each gate location must have one or two sites.")

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_svd(self, G_seq, where_seq, progbar=False, cutoff=1e-12, cutoff_mode="rsum2", fidelity_samples=10):
        """Sweep the gate stream with local ``reduce-split`` + left-compress.

        Two-site updates use ``apply_gate(..., contract='reduce-split')``
        followed by a canonicalise + left-compress sweep across the gate
        window down to bond ``chi``.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="svd",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                contract = (
                    "split-gate"
                    if self._has_symmray_data(p) and (xmax - xmin > 1)
                    else "reduce-split"
                )

                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=contract,
                    inplace=True,
                )

                self.canonize_mpo(p, (xmin, xmax))
                for i in range(xmax, xmin, -1):
                    p.right_canonize_site(i, bra=None)
                p.left_compress(
                    start=xmin,
                    stop=xmax,
                    max_bond=self.chi,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
            else:
                raise ValueError("Each gate location must have one or two sites.")

            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    def _run_mpo(self, G_seq, where_seq, progbar=False, cutoff=1e-12, cutoff_mode="rsum2", fidelity_samples=10):
        """Sweep the gate stream with :func:`gate_nonlocal_opt` compression.

        Two-site gates are routed through ``gate_nonlocal_opt`` independently
        on the upper (ket) and lower (bra) MPO families using
        ``method="direct"``.  One-site gates are applied directly via
        :meth:`_apply_gate_pair`.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mpo",
                leave=True,
                position=0,
                colour="GREEN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                g_k, g_b = self._prepare_gate_pair(
                    gate,
                    n_sites,
                    bra_gate=bra_gate,
                    p=p,
                    where=where,
                    ind_id=self.ind_id_k,
                )
                if g_k is not None:
                    p = gate_nonlocal_opt(
                        p, g_k, where,
                        which="upper", method="direct",
                        info=self.info_c, inplace=True,
                        ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                        max_bond=self.chi, cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                if g_b is not None:
                    p = gate_nonlocal_opt(
                        p, g_b, where,
                        which="lower", method="direct",
                        info=self.info_c, inplace=True,
                        ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                        max_bond=self.chi, cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                self.p = p
            else:
                raise ValueError("Each gate location must have one or two sites.")

            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(self.p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": self.p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=6,
        *,
        mode=None,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        fidelity_samples=10,
        k_2q_batch=1,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        fit_max_span="auto",
        fit_three_site_sweeps=1,
        target_cutoff=0.0,
    ):
        """Run queued gates on both MPO index families.

        Parameters
        ----------
        n_iter : int, default=6
            Inner iterations for DMRG ``FIT`` updates on two-site gates.
            Ignored by ``svd`` mode.
        mode : {"dmrg", "svd", "mpo"} | None, default=None
            Optional mode override for this run.
        progbar : bool, default=False
            Show tqdm progress bar.
        cutoff : float | {"auto"}, default=1e-12
            Truncation cutoff used in gate application and compression.
        cutoff_mode : str, default="rsum2"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``.
        fidelity_samples : int, default=10
            ``svd`` mode only: number of intermediate norm-proxy samples.
            A final sample is always recorded at the end of the run.
        k_2q_batch : int, default=1
            ``dmrg`` mode only: number of sequential two-qubit gates to batch
            into one local FIT update. The FIT window uses the batch-wide
            ``[xmin, xmax]`` from all gate locations in the batch.
        fit_block_size : {1, 2, 3}, default=2
            ``dmrg`` mode only: number of neighboring MPO tensors optimized
            by each native FIT update. Two- and three-site updates grow only
            bonds reached by their SVD splits; one-site FIT retains the legacy
            fixed-rank compatibility path.
        fit_sweep_sequence : str, default="RL"
            ``dmrg`` mode only: cyclic FIT sweep directions. ``"R"`` and
            ``"L"`` select one direction, while ``"RL"`` alternates them.
        fit_max_span : int | {"auto"} | None, default="auto"
            ``dmrg`` mode only: maximum inclusive spatial span of a batched
            FIT target. The first long-range gate is always retained.
        fit_three_site_sweeps : int, default=1
            ``dmrg`` mode only: initial three-site warm-up sweeps before
            one-site refinement when ``fit_block_size=3``.
        target_cutoff : float, default=0.0
            ``dmrg`` mode only: cutoff used while constructing the target
            MPO. The output FIT SVD remains controlled by ``cutoff``.

        Returns
        -------
        qtn.MatrixProductOperator
            Updated MPO after replaying the queued gate stream.

        Notes
        -----
        Symmray MPOs use the block-aware SVD compression implementation for
        all three modes. This preserves multi-sector bonds when Quimb's
        generic dense auxiliary or bond-padding paths are unavailable.
        """
        if mode is not None:
            self.set_mode(mode)

        cutoff = self._resolve_cutoff(cutoff, self.p)

        G_seq = list(self.G)
        where_seq = list(self.where)
        if not G_seq:
            return self.p

        if self.mode == "dmrg":
            if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {
                1,
                2,
                3,
            }:
                raise ValueError("fit_block_size must be 1, 2, or 3.")
            if (
                not isinstance(fit_three_site_sweeps, Integral)
                or int(fit_three_site_sweeps) < 1
            ):
                raise ValueError("fit_three_site_sweeps must be a positive integer.")
            if int(fit_block_size) != 3 and int(fit_three_site_sweeps) != 1:
                raise ValueError(
                    "fit_three_site_sweeps is only configurable when "
                    "fit_block_size=3."
                )
            fit_max_span = self._resolve_fit_max_span(
                fit_max_span,
                k_2q_batch,
            )
            target_cutoff = float(target_cutoff)
            if not np.isfinite(target_cutoff) or target_cutoff < 0.0:
                raise ValueError(
                    "target_cutoff must be a finite non-negative number."
                )
            # Native Symmray FIT uses block-aware target construction and
            # native SVD splits. It must not take Quimb's eager bond-padding
            # route, but it otherwise follows the same variational DMRG path
            # as dense MPOs. ``mode='mpo'`` remains the direct SVD path.
            self._prepare_dmrg_state(fit_block_size=fit_block_size)
            self._run_dmrg(
                G_seq,
                where_seq,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                k_2q_batch=k_2q_batch,
                fidelity_samples=fidelity_samples,
                fit_block_size=fit_block_size,
                fit_sweep_sequence=fit_sweep_sequence,
                fit_max_span=fit_max_span,
                fit_three_site_sweeps=int(fit_three_site_sweeps),
                target_cutoff=target_cutoff,
            )
            return self.p

        if self.mode == "svd":
            self._run_svd(
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
            )
            return self.p

        if self.mode == "mpo":
            # ``gate_nonlocal_opt`` creates a dense auxiliary sub-MPO and its
            # generic compression currently loses multi-sector Symmray bond
            # metadata. Reuse the block-aware local SVD route for these MPOs.
            if self._has_symmray_data(self.p):
                self._run_svd(
                    G_seq,
                    where_seq,
                    progbar=progbar,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    fidelity_samples=fidelity_samples,
                )
                return self.p
            self._run_mpo(
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
            )
            return self.p

        supported = ", ".join(sorted(self._ALLOWED_MODES))
        raise ValueError(f"Unknown mode: {self.mode}. Supported modes: {supported}")

    def canonize_mpo(self, p, where):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``.  Integers and singletons collapse to a single-site
        orthogonality center.
        """
        if isinstance(where, Integral):
            site = int(where)
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 1:
            site = int(where[0])
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 2:
            xmin, xmax = min(int(where[0]), int(where[1])), max(int(where[0]), int(where[1]))
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        p.canonize(where_canon, cur_orthog=self._current_orthog(p))
        self.info_c["cur_orthog"] = target_orthog

    def get_fidelities(self):
        """Return the running loss history."""
        return self.losses

    def compress(self, *, cutoff=1e-12, cutoff_mode="rsum2"):
        """Compress the current MPO to ``chi`` while preserving its backend.

        Symmray MPOs use Quimb's local block-aware compression path. This is
        also useful when the optimizer is constructed with an empty gate
        queue and the caller only wants a bond-dimension reduction.
        """
        self.p.compress(
            form="left",
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        self._init_canonicalization()
        return self.p
