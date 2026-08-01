"""Native tree-tensor-network energy measurement and optimization.

This mirrors the energy surface of
:class:`~pepsy.optimizers.energy.MpsEnergyOptimizer` for a
:class:`~pepsy.optimizers.tree.TreeTensorNetwork`.  The energy is the sum of
per-term local expectations ``<psi|H_i|psi>/<psi|psi>`` evaluated with the
tree's own graded (fermion-safe) contraction, reusing a single contraction
optimiser and the cached norm across every term.  Optimization uses Quimb's
``TNOptimizer`` over the tree tensors while retaining that same exact local
expectation objective.  The public measurement path can cache the norm, but
the autodiff optimization path recomputes the full norm on every call: Quimb
injects tensor parameters directly and that mutation is outside the TTN's
cache-invalidation hooks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import autoray as ar

from ...backends import infer_backend_converter_from_sample
from ...tensors import build_optimizer
from .peps import EnergyEstimate, PepsEnergyOptimizer

__all__ = ["TreeEnergyOptimizer"]


class TreeEnergyOptimizer(PepsEnergyOptimizer):
    """Measure and optimize term-by-term energy on a tree state.

    The objective is the normalized local expectation
    ``sum_i <psi|H_i|psi> / <psi|psi>``.  Each local term is evaluated with the
    tree's exact :meth:`~pepsy.optimizers.tree.TreeTensorNetwork.local_expectation`
    path, which keeps native fermionic / U1 / U1U1 / Z2 Symmray operators
    structured and contracts the complete doubled tree so graded boundary phases
    are never discarded.  Terms are dispatched through
    :meth:`~pepsy.optimizers.tree.TreeTensorNetwork.local_expectations`, which
    shares one contraction optimiser and the memoized graded norm across the
    whole batch.

    Provide the local terms as a ``{where: operator}`` mapping via ``terms=`` or
    as a Pepsy symmetric Hamiltonian exposing such a mapping via ``.terms``
    (``hamiltonian=`` is accepted as a compatibility alias).  Each ``where`` is
    an int tree site or a tuple of tree sites.  When ``contraction_opt`` is a
    plain string a reusable :func:`pepsy.build_optimizer` is built once so paths
    are cached across terms and across repeated calls; pass an explicit reusable
    optimiser object to control caching directory / parallelism.
    """

    _LOSS_KEYS = frozenset({
        "normalized",
        "energy_per_site",
        "real",
        "contraction_opt",
    })

    def __init__(
        self,
        state,
        hamiltonian=None,
        *,
        terms=None,
        normalized: bool = True,
        energy_per_site: bool = True,
        real: bool = True,
        contraction_opt: Any = "auto-hq",
        loss_kwargs: Mapping[str, Any] | None = None,
    ):
        if hamiltonian is not None and terms is not None:
            raise TypeError("pass either hamiltonian or terms, not both")
        source = terms if terms is not None else hamiltonian
        self.state = self._as_tree_state(state)
        self.hamiltonian = source
        self.terms = self._terms_from_hamiltonian(source)
        self.loss_kwargs = {
            "normalized": bool(normalized),
            "energy_per_site": bool(energy_per_site),
            "real": bool(real),
            "contraction_opt": self._resolve_optimize(contraction_opt),
        }
        self.losses: list[float] = []
        if loss_kwargs is not None:
            self.set_loss_kwargs(**dict(loss_kwargs))

    # -- state / term / option resolution -------------------------------------

    @staticmethod
    def _as_tree_state(state):
        if hasattr(state, "local_expectations") and hasattr(state, "plan"):
            return state
        for name in ("tn", "p"):
            inner = getattr(state, name, None)
            if inner is not None and hasattr(inner, "local_expectations"):
                return inner
        raise TypeError(
            "state must be a TreeTensorNetwork with local_expectations()."
        )

    # PepsEnergyOptimizer's shared optimization helpers use this hook for
    # state validation. Keep the tree-specific validation while reusing its
    # backend conversion and finite-gradient machinery.
    _as_peps_state = staticmethod(_as_tree_state)

    @staticmethod
    def _terms_from_hamiltonian(source):
        if source is None:
            raise ValueError("a Hamiltonian or terms mapping is required.")
        terms = source.terms if hasattr(source, "terms") else source
        if not isinstance(terms, Mapping):
            raise TypeError(
                "terms must be a mapping {where: operator} or a Hamiltonian "
                "exposing such a mapping via `.terms`."
            )
        return dict(terms)

    @staticmethod
    def _resolve_optimize(contraction_opt):
        # A plain string re-plans on every contraction; a reusable optimiser
        # caches one path per contraction topology, which is the whole point of
        # batching the terms.  Build one lazily so the default already reuses.
        if isinstance(contraction_opt, str):
            return build_optimizer(progbar=False)
        return contraction_opt

    @classmethod
    def _pick_loss_kwargs(cls, options):
        incoming = dict(options or {})
        unknown = sorted(set(incoming) - cls._LOSS_KEYS)
        if unknown:
            allowed = ", ".join(sorted(cls._LOSS_KEYS))
            raise TypeError(
                f"Unknown tree energy option(s): {', '.join(unknown)}. "
                f"Allowed options: {allowed}."
            )
        if "contraction_opt" in incoming:
            incoming["contraction_opt"] = cls._resolve_optimize(
                incoming["contraction_opt"]
            )
        return incoming

    @staticmethod
    def _num_sites(state):
        num_sites = getattr(state, "nsites", None)
        if num_sites is not None:
            return int(num_sites)
        sites = getattr(state, "sites", None)
        if sites is not None:
            return len(tuple(sites))
        raise ValueError("Could not infer the number of tree sites.")

    @staticmethod
    def _max_bond(state):
        max_bond = getattr(state, "max_bond", None)
        if callable(max_bond):
            try:
                value = max_bond()
            except Exception:  # pragma: no cover - defensive for odd states
                return None
            return None if value is None else int(value)
        return None

    @staticmethod
    def _maybe_real(value):
        try:
            return ar.do("real", value)
        except Exception:  # pragma: no cover - defensive for scalar types
            return getattr(value, "real", value)

    @classmethod
    def _terms_for_state_backend(cls, terms, state):
        """Convert dense operator terms to the live tree backend."""
        sample = cls._sample_array_from_tn(state)
        try:
            converter = infer_backend_converter_from_sample(sample)
        except (ImportError, TypeError, ValueError):
            converter = None
        if converter is None:
            return terms
        return {
            where: cls._convert_term_array(operator, converter)
            for where, operator in dict(terms).items()
        }

    @classmethod
    def _loss_state(
        cls,
        state,
        *,
        terms,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
    ):
        """Evaluate the differentiable tree energy objective."""
        state = cls._as_tree_state(state)
        terms = cls._terms_from_hamiltonian(terms)
        terms = cls._terms_for_state_backend(terms, state)
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)
        values = state.local_expectations(
            terms,
            optimize=contraction_opt,
            normalized=bool(normalized),
        )
        values = tuple(values.values())
        total = 0.0 if not values else values[0]
        for value in values[1:]:
            total = total + value
        if real:
            total = cls._maybe_real(total)
        if energy_per_site:
            total = total / cls._num_sites(state)
        return total

    @classmethod
    def _optimization_loss_state(
        cls,
        state,
        *,
        terms,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
    ):
        """Evaluate the optimization objective with a fresh norm.

        ``TNOptimizer`` updates tensor arrays through ``apply_to_arrays``.
        That is deliberately a low-level operation and cannot invalidate the
        TTN's memoized native-fermion norm (or its canonical-region metadata).
        Calling ``local_expectation(..., normalized=True)`` here would
        therefore divide a newly injected state by the previous state's norm.
        Compute every term unnormalized, then divide by a fresh full doubled
        tree contraction.  This is the gauge-invariant Rayleigh quotient and
        remains differentiable through the live tensor backend.
        """
        state = cls._as_tree_state(state)
        terms = cls._terms_from_hamiltonian(terms)
        terms = cls._terms_for_state_backend(terms, state)
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)

        values = tuple(
            state.local_expectation(
                operator,
                where,
                optimize=contraction_opt,
                normalized=False,
            )
            for where, operator in terms.items()
        )
        total = 0.0 if not values else values[0]
        for value in values[1:]:
            total = total + value
        if normalized:
            denominator = (state.H | state).contract(
                all,
                optimize=contraction_opt,
            )
            total = total / denominator
        if real:
            total = cls._maybe_real(total)
        if energy_per_site:
            total = total / cls._num_sites(state)
        return total

    @staticmethod
    def _tnopt_loss(state, *, terms, **loss_kwargs):
        """Adapter for :class:`quimb.tensor.TNOptimizer`."""
        return TreeEnergyOptimizer._optimization_loss_state(
            state,
            terms=terms,
            **loss_kwargs,
        )

    # -- energy evaluation ----------------------------------------------------

    def _total_energy(self, state, terms, opts):
        return self._loss_state(
            state,
            terms=terms,
            normalized=opts["normalized"],
            energy_per_site=False,
            real=opts["real"],
            contraction_opt=opts["contraction_opt"],
        )

    def _resolve(self, state, hamiltonian, terms, kwargs):
        state = self.state if state is None else self._as_tree_state(state)
        terms_use = self.terms
        if hamiltonian is not None:
            terms_use = self._terms_from_hamiltonian(hamiltonian)
        if terms is not None:
            terms_use = self._terms_from_hamiltonian(terms)
        opts = dict(self.loss_kwargs)
        opts.update(self._pick_loss_kwargs(kwargs))
        return state, terms_use, opts

    def loss(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Return the scalar energy (per-site when ``energy_per_site``)."""
        state, terms_use, opts = self._resolve(state, hamiltonian, terms, kwargs)
        total = self._total_energy(state, terms_use, opts)
        if opts["energy_per_site"]:
            return total / self._num_sites(state)
        return total

    def energy(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Return full and per-site energy estimates for ``state``."""
        state, terms_use, opts = self._resolve(state, hamiltonian, terms, kwargs)
        energy = self._total_energy(state, terms_use, opts)
        num_sites = self._num_sites(state)
        energy_per_site = energy / num_sites
        return EnergyEstimate(
            energy=energy,
            energy_per_site=energy_per_site,
            num_sites=num_sites,
            chi=self._max_bond(state),
            boundary_mode="exact",
            normalized=bool(opts["normalized"]),
            metadata={
                "real": opts["real"],
                "contraction_opt": opts["contraction_opt"],
            },
        )

    def set_loss_kwargs(self, **kwargs):
        """Update the defaults used by :meth:`loss` and :meth:`optimize`."""
        self.loss_kwargs.update(self._pick_loss_kwargs(kwargs))
        return self

    def normalize(self, state=None, **kwargs):
        """Normalize a tree state in place using its canonical norm path."""
        state = self.state if state is None else self._as_tree_state(state)
        normalize = getattr(state, "normalize", None)
        if not callable(normalize):
            raise TypeError("state must provide normalize() for tree normalization.")
        normalize(**kwargs)
        return state

    @staticmethod
    def _recanonicalize(state):
        """Invalidate metadata after direct TN parameter updates.

        Arbitrary tensor-array updates are not guaranteed to admit an exact
        canonicalization through the old centre.  In particular, doing so for
        native fermionic arrays can create a centre-norm shortcut inconsistent
        with the updated state.  Leave the post-optimization state
        non-canonical and force subsequent normalized fermionic readouts down
        the exact full-network norm path.
        """
        plan = getattr(state, "plan", None)
        node_tensor = getattr(state, "node_tensor", None)
        if plan is not None and callable(node_tensor):
            # TNOptimizer updates tensor data directly. Clear Quimb's local
            # ``left_inds`` proofs as well as the TTN's region marker, or a
            # subsequent canonicalization can incorrectly trust orientations
            # inherited from the pre-optimization state.
            for node in plan.nodes():
                node_tensor(node).modify(left_inds=None)
        invalidate = getattr(state, "invalidate_canonical_form", None)
        if callable(invalidate):
            invalidate()
        return state

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a Quimb ``TNOptimizer`` for the live tree state.

        The shared PEPS implementation supplies backend conversion and the
        Quimb optimizer construction; this override exists as the explicit
        tree API and documents that the objective is the tree loss adapter.
        """
        return super().make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
            **tnopt_kwargs,
        )

    def optimize(
        self,
        *,
        n=220,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        return_losses: bool = False,
        normalize: bool = False,
        normalize_kwargs: Mapping[str, Any] | None = None,
        check_finite_gradient: bool = True,
        **optimize_kwargs,
    ):
        """Optimize the tree tensors against the configured energy objective."""
        merged_loss_kwargs = dict(self.loss_kwargs)
        merged_loss_kwargs.update(self._pick_loss_kwargs(loss_kwargs))
        tnopt = self.make_tn_optimizer(
            loss_kwargs=merged_loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        if check_finite_gradient:
            finite_gradient, finite_loss = self._initial_gradient_status(tnopt)
            if not finite_gradient:
                self.losses = [] if finite_loss is None else [float(finite_loss)]
                warnings.warn(
                    "Tree energy autodiff produced a non-finite initial "
                    "gradient; returning the unmodified state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if return_losses:
                    return self.state, tuple(self.losses)
                return self.state

        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        out = self._state_for_autodiff_backend(
            out,
            self.state,
            autodiff_backend,
            device=device,
        )
        out = self._recanonicalize(self._as_tree_state(out))
        if normalize:
            out = self.normalize(out, **dict(normalize_kwargs or {}))
        self.state = self._as_tree_state(out)
        if return_losses:
            return self.state, tuple(self.losses)
        return self.state
