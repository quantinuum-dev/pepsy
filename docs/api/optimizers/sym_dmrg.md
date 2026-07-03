# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`.

For Symmray Fermi-Hubbard MPOs, the first Pepsy-native slice now builds dense
left/right environments for `<psi|MPO|psi>`, checks their energy against the
existing `MpsEnergyOptimizer` MPO path, and exposes a sector-preserving two-site
matvec whose active basis is exactly the current `theta` block layout. The
local solver is an exact dense reference solve and is enabled for `L=2`
correctness runs; longer-chain sweeps still stop until the canonical-center /
effective-norm update is added.

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
