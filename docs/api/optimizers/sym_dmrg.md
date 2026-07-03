# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`. Symmray
MPOs currently initialize the Pepsy block-sparse path and record charge-sector
and initial-energy diagnostics; the sector-preserving local eigensolver is the
next implementation step.

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
