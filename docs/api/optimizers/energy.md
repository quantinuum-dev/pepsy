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
