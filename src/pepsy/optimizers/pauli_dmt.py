"""Local open-chain operator TEBD with protected Pauli-MPO truncation."""

from __future__ import annotations

from numbers import Integral

import autoray as ar
import numpy as np

from ..operators._pauli_dmt import (
    _PauliDMTWorkspace,
    _dag,
    _gate_transfer,
)
from ..operators.pauli_mpo import (
    PauliMPO,
    _array_like,
    _complexify,
    _core_bond_dimensions,
    _host_array,
    _normalize_where,
)

__all__ = ["PauliDMT"]


class PauliDMT:
    """Persist canonical and trace environments across local DMT updates.

    ``variant='density'`` with ``mode='conjugate'`` evolves a density matrix.
    ``variant='operator'`` with ``mode='heisenberg'`` evolves an observable,
    including traceless infinite-temperature perturbations. Input is copied;
    :attr:`state` returns an independent snapshot. Reports cover the latest
    gate or final timestep; no unbounded trajectory history is retained.

    Only adjacent increasing sites on an open chain are supported. An
    evolution call commits successful gates as it proceeds, so after a
    numerical exception the state can contain a completed partial timestep.
    ``trace_guard='auto'`` monitors per-gate relative trace defects for
    density inputs and operator inputs whose trace appears resolved.
    ``trace_guard=True`` forces it and ``False`` disables it; inspect
    :attr:`trace_guard_active` for the resolved choice. This guard does not
    certify all local marginals; monitor observables and convergence.
    ``cutoff='auto'`` uses Pepsy's dtype-aware cutoff; ``cutoff_mode``
    follows Quimb and applies only to the unprotected residual.
    ``regions`` replaces contiguous three-site protection with complete
    reduced-operator constraints on the given MPO-site sequences. Geometry-
    selected splits check those regions before committing each update.
    """

    def __init__(self, operator, *, max_bond, cutoff=0.0, cutoff_mode="rel", variant="density",
                 mode="conjugate", diagnostics=False, trace_guard="auto", regions=None):
        if not isinstance(operator, PauliMPO):
            raise TypeError("operator must be a PauliMPO.")
        if mode not in {"conjugate", "heisenberg"}:
            raise ValueError("mode must be 'conjugate' or 'heisenberg'.")
        self.mode = mode
        self.time = 0.0
        self._workspace = _PauliDMTWorkspace(
            operator, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant,
            diagnostics=diagnostics, trace_guard=trace_guard, regions=regions,
        )

    @property
    def state(self):
        """An independent PauliMPO snapshot, retaining backend and gradients."""
        state = PauliMPO.from_pauli_cores(self._workspace.cores)
        state.compression_report = self.report
        return state

    @property
    def report(self):
        return self._workspace.report()

    @property
    def trace_guard_active(self):
        """Whether gated updates monitor relative trace defects."""
        return self._workspace.monitor_trace

    def _begin(self):
        self._workspace.original_bonds = _core_bond_dimensions(self._workspace.cores)
        self._workspace.reports.clear()

    def apply_gate(self, gate, where):
        """Apply one two-qubit gate and protected split; return its report."""
        where = _normalize_where(where, len(self._workspace.cores))
        if len(where) != 2 or where[1] != where[0] + 1:
            raise ValueError("DMT gates require two adjacent sites in increasing order.")
        transfer = _gate_transfer(gate, self._workspace.cores[0], self.mode)
        self._begin()
        return self._workspace.split_pair(where[0], transfer)

    def evolve(self, bond_hamiltonians, *, dt, steps=1):
        """Second-order even/odd/even TEBD for ``H=sum_j h[j]``.

        Supply exactly ``L-1`` Hermitian 4-by-4 matrices in increasing bond
        order. Include onsite fields in these matrices without double counting.
        Gates/transfers are prepared once per call with backend-native EIGH.
        Shared Hamiltonian objects reuse preparation within this call only.
        Parity layers sweep from the nearest end to carry the canonical
        centre across timesteps; this fixes the finite-bond compression order.
        Timestep error and compression error remain separate. Returns an
        independent final state; the live workspace can continue evolving.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")
        hs = tuple(bond_hamiltonians)
        n = len(self._workspace.cores)
        if len(hs) != n - 1:
            raise ValueError("supply exactly L-1 bond Hamiltonians.")
        sample = _complexify(self._workspace.cores[0])
        transfers, eigensystems, prepared = [], {}, {}
        for bond, h in enumerate(hs):
            # Shared Hamiltonian objects are common in uniform chains. Cache
            # only within this call: cross-call reuse could retain stale array
            # values or a consumed Torch autograd graph.
            key = id(h)
            if key not in eigensystems:
                if tuple(ar.shape(h)) != (4, 4):
                    raise ValueError("each bond Hamiltonian must have shape (4, 4).")
                native_h = _array_like(h, sample)
                host = _host_array(native_h)
                tolerance = 64 * np.finfo(host.real.dtype).eps
                magnitude = np.max(np.abs(host))
                scaled = host / magnitude if np.isfinite(magnitude) and magnitude else host
                if not np.all(np.isfinite(host)) or not np.allclose(
                    scaled, scaled.conj().T, rtol=tolerance, atol=tolerance,
                ):
                    raise ValueError("bond Hamiltonians must be finite and Hermitian.")
                eigensystems[key] = ar.do("linalg.eigh", native_h)
            fraction = 0.5 if bond % 2 == 0 else 1.0
            if (key, fraction) not in prepared:
                w, v = eigensystems[key]
                gate = (v * ar.do("exp", (-1j * dt * fraction) * w)) @ _dag(v)
                prepared[key, fraction] = _gate_transfer(gate, self._workspace.cores[0], self.mode)
            transfers.append(prepared[key, fraction])
        self._begin()
        if dt == 0.0 or steps == 0:
            return self.state
        even, odd = range(0, n - 1, 2), range(1, n - 1, 2)
        for _ in range(steps):
            # Start at the nearest end, then turn at each layer. Carrying
            # the centre across timesteps avoids a full-chain return sweep.
            # Disjoint gates commute exactly; with DMT their compression
            # order is explicitly this centre-aware serpentine schedule.
            forward = not even or self._workspace.center <= (even[-1] + 1) / 2
            outer = even if forward else reversed(even)
            inner = reversed(odd) if forward else odd
            absorb = "right" if forward else "left"
            for bond in outer:
                self._workspace.split_pair(bond, transfers[bond], absorb=absorb)
            for bond in inner:
                self._workspace.split_pair(bond, transfers[bond],
                                           absorb="left" if forward else "right")
            for bond in even if forward else reversed(even):
                self._workspace.split_pair(bond, transfers[bond], absorb=absorb)
            self.time += dt
            # Bound diagnostic memory by one timestep, independent of steps.
            if _ + 1 != steps:
                self._workspace.reports.clear()
        return self.state
