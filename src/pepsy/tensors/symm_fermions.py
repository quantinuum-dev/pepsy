"""Curated public helpers for native symmetric fermion workflows.

The underlying Symmray adapters live in :mod:`pepsy.tensors.symmetric`.  This
module is the model-facing home for convenient fermionic building blocks, so
future spinless or multicomponent helpers can live alongside ``SpinfulFermion``
without making the symmetry implementation module into a model catalogue.
"""

from .symmetric import SpinfulFermion

__all__ = ["SpinfulFermion", "SpinfulFermionHubbard", "SymmFermions"]


class SymmFermions:
    """Namespace for direct Symmray-backed fermion helper factories."""

    @staticmethod
    def spinful(*args, **kwargs):
        """Build a :class:`SpinfulFermion` helper.

        This namespace makes it possible to add other local fermion spaces
        later while keeping the direct ``SpinfulFermion(...)`` form concise.
        """
        return SpinfulFermion(*args, **kwargs)


# Compatibility spelling for the initial, overly model-specific public name.
SpinfulFermionHubbard = SpinfulFermion
