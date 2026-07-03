# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`.

For Symmray Fermi-Hubbard MPOs, Pepsy builds dense left/right environments for
`<psi|MPO|psi>`, exposes sector-preserving two-site `H_eff` and `N_eff` matvecs
whose active basis is exactly the current `theta` block layout, and can solve
the local problem either with a dense reference matrix or a quimb-compatible
Lanczos `LinearOperator`. Symmray sweeps canonicalize the MPS center before
using H-only dense/Lanczos solves and fall back to the explicit dense
generalized problem when the effective norm is not identity-like.

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
