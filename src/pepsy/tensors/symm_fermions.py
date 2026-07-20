"""Curated public helpers for native symmetric fermion workflows.

The underlying Symmray adapters live in :mod:`pepsy.tensors.symmetric`.  This
module is the model-facing home for convenient fermionic building blocks, so
future spinless or multicomponent helpers can live alongside ``SpinfulFermion``
without making the symmetry implementation module into a model catalogue.
"""

from .symmetric import Fermion

SpinfulFermion = Fermion
SpinfulFermionHubbard = Fermion

__all__ = [
    "Fermion",
    "SpinfulFermion",
    "SpinfulFermionHubbard",
    "SymmFermions",
]


class SymmFermions:
    """Namespace for direct Symmray-backed fermion helper factories."""

    @staticmethod
    def fermion(*args, **kwargs):
        """Build the unified :class:`Fermion` helper."""
        return Fermion(*args, **kwargs)

    @staticmethod
    def spinful(*args, **kwargs):
        """Build a spinful :class:`Fermion` helper.

        This namespace makes it possible to add other local fermion spaces
        later while keeping the direct ``SpinfulFermion(...)`` form concise.
        """
        return Fermion(*args, **kwargs)

    @staticmethod
    def spinless(*args, **kwargs):
        """Build a spinless :class:`Fermion` helper."""
        kwargs.setdefault("spinful", False)
        kwargs.setdefault("symmetry", "U1")
        return Fermion(*args, **kwargs)


# Compatibility spelling for the initial, overly model-specific public name.
