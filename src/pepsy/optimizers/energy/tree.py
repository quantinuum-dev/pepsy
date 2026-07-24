"""Native tree-tensor-network energy measurement.

This mirrors the *measurement* surface of
:class:`~pepsy.optimizers.energy.MpsEnergyOptimizer` for a
:class:`~pepsy.optimizers.tree.TreeTensorNetwork`.  The energy is the sum of
per-term local expectations ``<psi|H_i|psi>/<psi|psi>`` evaluated with the
tree's own graded (fermion-safe) contraction, reusing a single contraction
optimiser and the cached norm across every term.  It is deliberately thin: the
tree gate-stream evolution/optimization lives in
:class:`~pepsy.optimizers.tree.TreeOptimizer`; this class only reports energy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import autoray as ar

from ...tensors import build_optimizer
from .peps import EnergyEstimate

__all__ = ["TreeEnergyOptimizer"]


class TreeEnergyOptimizer:
    """Report term-by-term local energy for a :class:`TreeTensorNetwork`.

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

    # -- state / term / option resolution -------------------------------------

    @staticmethod
    def _as_tree_state(state):
        if hasattr(state, "local_expectations") and hasattr(state, "plan"):
            return state
        inner = getattr(state, "p", None)
        if inner is not None and hasattr(inner, "local_expectations"):
            return inner
        raise TypeError(
            "state must be a TreeTensorNetwork with local_expectations()."
        )

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

    # -- energy evaluation ----------------------------------------------------

    def _total_energy(self, state, terms, opts):
        values = state.local_expectations(
            terms,
            optimize=opts["contraction_opt"],
            normalized=bool(opts["normalized"]),
        )
        total = sum((complex(v) for v in values.values()), 0j)
        if opts["real"]:
            return self._maybe_real(total)
        return total

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
