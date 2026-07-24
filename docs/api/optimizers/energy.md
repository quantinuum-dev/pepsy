# `pepsy.optimizers.energy`

```{eval-rst}
.. automodule:: pepsy.optimizers.energy
   :members:
```

## Symmray fermionic encodings

For a native fermionic Symmray MPS, prefer native local Hamiltonian terms:

```python
import pepsy

estimate = pepsy.MpsEnergyOptimizer(
    fermionic_state,
    terms=hamiltonian,
    energy_per_site=False,
).energy()
```

``terms=`` accepts either a local-term mapping or the complete
``SymHamiltonian`` returned by ``fermion.hamiltonian(terms)``. For an explicit
one-site-plus-two-site Hamiltonian, passing the ``SymHamiltonian`` preserves
the fermionic ordering metadata and lets a native fermionic MPS evaluate the
terms directly without constructing the bosonic MPO.

The MPO returned by a Fermi-Hubbard ``SymHamiltonian.to_mpo(...)`` uses the
bosonic/Jordan-Wigner representation required by the DMRG MPO path. Pepsy now
rejects an implicit contraction of that MPO with a native fermionic MPS, since
the automatic re-encoding can create very large block-sparse intermediates.
For a deliberately small or explicitly managed conversion, pass
``allow_encoding_conversion=True`` to ``MpsEnergyOptimizer``.

## Tree tensor networks

``TreeEnergyOptimizer`` mirrors the ``MpsEnergyOptimizer`` measurement surface
for a :class:`~pepsy.optimizers.tree.TreeTensorNetwork`. It reports
``sum_i <psi|H_i|psi> / <psi|psi>`` term by term using the tree's own exact,
fermion-safe contraction, and returns the same :class:`EnergyEstimate`:

```python
import pepsy

estimate = pepsy.TreeEnergyOptimizer(
    tree_state,
    terms=hamiltonian,        # {where: operator} mapping or a SymHamiltonian
    energy_per_site=True,
).energy()
```

The terms are dispatched through
:meth:`~pepsy.optimizers.tree.TreeTensorNetwork.local_expectations`, which
shares one contraction optimiser across every term (pass a reusable
``pepsy.build_optimizer(...)`` as ``contraction_opt`` to cache paths across
same-topology contractions) and reuses the memoized graded norm, so the
result is identical to summing per-term
:meth:`~pepsy.optimizers.tree.TreeTensorNetwork.local_expectation` calls.
