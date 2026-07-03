"""Symmetry-aware two-site DMRG driver.

This module provides the public :class:`SymDMRG2` API that Pepsy will grow into
for Symmray-backed block-sparse Hamiltonians. Ordinary quimb MPOs are delegated
directly to :class:`quimb.tensor.DMRG2`; Symmray MPOs currently initialize the
sector-aware driver state and energy bookkeeping, with the local block-sparse
two-site eigensolver left as the next implementation slice.
"""

from __future__ import annotations

from .energy import MpsEnergyOptimizer


def _is_symmray_array(value):
    return type(value).__module__.split(".", 1)[0] == "symmray"


def _unwrap_state(state):
    if state is None:
        return None
    if hasattr(state, "tn"):
        return state.tn
    if hasattr(state, "psi"):
        return state.psi
    return state


def _iter_tensor_data(obj):
    obj = _unwrap_state(obj)
    if obj is None:
        return
    if hasattr(obj, "tensor_map"):
        tensors = obj.tensor_map.values()
    else:
        try:
            tensors = tuple(obj)
        except TypeError:
            tensors = ()
    for tensor in tensors:
        yield getattr(tensor, "data", tensor)


def _uses_symmray_arrays(*objects):
    return any(
        _is_symmray_array(data)
        for obj in objects
        for data in _iter_tensor_data(obj)
    )


def _infer_total_charge(state):
    if state is None:
        return None
    overall_charge = getattr(state, "overall_charge", None)
    if callable(overall_charge):
        return overall_charge()
    return getattr(state, "total_charge", None)


def _normalize_backend(backend):
    key = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "quimb": "quimb",
        "quimb_dmrg": "quimb",
        "quimb_dmrg2": "quimb",
        "dense": "quimb",
        "symmray": "symmray",
        "pepsy": "symmray",
        "block_sparse": "symmray",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown SymDMRG2 backend {backend!r}. Expected one of: {allowed}.") from exc


class SymDMRG2:
    """Two-site DMRG facade for dense quimb and Symmray MPOs.

    Parameters
    ----------
    mpo
        Hamiltonian MPO. Dense/quimb MPOs are solved by delegating to
        ``quimb.tensor.DMRG2``. Symmray MPOs select the Pepsy block-sparse path.
    init_mps
        Optional initial MPS. Pepsy ``SymMPS`` wrappers and raw quimb MPS objects
        are both accepted.
    chi
        Maximum MPS bond dimension used for two-site splits.
    cutoff
        SVD truncation cutoff.
    sweeps
        Default number of DMRG sweeps for :meth:`solve`.
    total_charge
        Fixed global charge sector. If omitted, Pepsy tries to infer this from
        ``init_mps.overall_charge()``.
    backend
        ``"auto"`` selects ``"symmray"`` when either input carries Symmray array
        data, otherwise ``"quimb"``.
    which
        Quimb eigensolver target, e.g. ``"SA"`` for smallest algebraic.
    tol
        Default energy convergence tolerance for :meth:`solve`.
    dmrg_opts
        Advanced quimb DMRG options copied into ``driver.opts`` before solving.
    """

    def __init__(
        self,
        mpo,
        init_mps=None,
        *,
        chi=None,
        cutoff=1e-8,
        sweeps=4,
        total_charge=None,
        backend="auto",
        which="SA",
        tol=1e-4,
        dmrg_opts=None,
    ):
        if chi is None:
            chi = 32
        if int(chi) < 1:
            raise ValueError("chi must be a positive integer.")
        if int(sweeps) < 1:
            raise ValueError("sweeps must be a positive integer.")

        self.mpo = mpo
        self.init_mps = init_mps
        self.mps = _unwrap_state(init_mps)
        self.chi = int(chi)
        self.cutoff = float(cutoff)
        self.sweeps = int(sweeps)
        self.total_charge = total_charge if total_charge is not None else _infer_total_charge(init_mps)
        self.which = which
        self.tol = float(tol)
        self.dmrg_opts = {} if dmrg_opts is None else dict(dmrg_opts)

        requested_backend = _normalize_backend(backend)
        self.uses_symmray = _uses_symmray_arrays(mpo, init_mps)
        if requested_backend == "auto":
            self.backend = "symmray" if self.uses_symmray else "quimb"
        else:
            self.backend = requested_backend

        if self.backend == "quimb" and self.uses_symmray:
            raise ValueError(
                "backend='quimb' delegates to quimb.tensor.DMRG2 and is only "
                "enabled for ordinary dense/quimb MPOs. Use backend='symmray' "
                "for the Pepsy block-sparse path."
            )

        self.driver = None
        self.converged = None
        self.energies = []
        self._state = self.mps
        self.initial_energy = self._compute_initial_energy()

    @property
    def state(self):
        """Current optimized state, or the initial state before solving."""
        return self._state

    @property
    def energy(self):
        """Most recent sweep energy, falling back to the initial energy."""
        if self.energies:
            return self.energies[-1]
        return self.initial_energy

    def _compute_initial_energy(self):
        if self.init_mps is None:
            return None
        try:
            estimate = MpsEnergyOptimizer(
                self.init_mps,
                self.mpo,
                energy_per_site=False,
                real=False,
            ).energy()
        except Exception:  # pragma: no cover - best-effort diagnostic only
            return None
        return estimate.energy

    def _solve_quimb(
        self,
        *,
        chi,
        cutoff,
        sweeps,
        tol,
        verbosity,
        sweep_sequence,
        solve_opts,
    ):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        self.driver = qtn.DMRG2(
            self.mpo,
            which=self.which,
            bond_dims=chi,
            cutoffs=cutoff,
            p0=self.mps,
        )
        if self.dmrg_opts:
            self.driver.opts.update(self.dmrg_opts)

        kwargs = dict(solve_opts)
        if sweep_sequence is not None:
            kwargs["sweep_sequence"] = sweep_sequence
        self.converged = bool(
            self.driver.solve(
                tol=tol,
                bond_dims=chi,
                cutoffs=cutoff,
                max_sweeps=sweeps,
                verbosity=verbosity,
                **kwargs,
            )
        )
        self.energies = list(self.driver.energies)
        self._state = self.driver.state
        return self

    def _solve_symmray(self):
        raise NotImplementedError(
            "Symmray DMRG2 local eigensolver is not implemented yet. "
            "This scaffold already fixes the API, total-charge bookkeeping, "
            "and initial MPO-energy diagnostic; the next slice is the "
            "block-sparse two-site effective Hamiltonian and Lanczos/SVD update."
        )

    def solve(
        self,
        *,
        tol=None,
        sweeps=None,
        chi=None,
        cutoff=None,
        verbosity=0,
        sweep_sequence=None,
        **solve_opts,
    ):
        """Run DMRG2 and return ``self``.

        Dense/quimb MPOs are solved immediately by quimb's implementation.
        Symmray MPOs currently raise :class:`NotImplementedError` at solve time,
        after initialization has recorded charge-sector and initial-energy
        diagnostics.
        """
        chi = self.chi if chi is None else int(chi)
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        sweeps = self.sweeps if sweeps is None else int(sweeps)
        tol = self.tol if tol is None else float(tol)

        if chi < 1:
            raise ValueError("chi must be a positive integer.")
        if sweeps < 1:
            raise ValueError("sweeps must be a positive integer.")

        if self.backend == "quimb":
            return self._solve_quimb(
                chi=chi,
                cutoff=cutoff,
                sweeps=sweeps,
                tol=tol,
                verbosity=verbosity,
                sweep_sequence=sweep_sequence,
                solve_opts=solve_opts,
            )
        return self._solve_symmray()

    run = solve

    def summary(self):
        """Return lightweight setup and progress metadata."""
        return {
            "backend": self.backend,
            "uses_symmray": self.uses_symmray,
            "chi": self.chi,
            "cutoff": self.cutoff,
            "sweeps": self.sweeps,
            "total_charge": self.total_charge,
            "initial_energy": self.initial_energy,
            "energy": self.energy,
            "converged": self.converged,
        }
