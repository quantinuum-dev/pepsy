# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`.

For Symmray Fermi-Hubbard MPOs, Pepsy assumes an OBC MPS/MPO chain. Periodic
lattice edges should be encoded as long-range terms in that OBC MPO, not as a
cyclic MPS. Pepsy builds dense left/right environments for `<psi|MPO|psi>`,
block-sparse environments for the projected `H_eff`, and dense debug
environments for `N_eff`. The active local basis is exactly the current
`theta` block layout. By default, Symmray `H_eff` matvecs use a block-native
projected contraction; `matvec_backend="dense_reference"` keeps the older
NumPy dense-aligned matvec available as a validator. Symmray sweeps
canonicalize the MPS center before using H-only dense/Lanczos solves; a
non-identity effective norm is treated as a canonicalization/alignment error
unless the explicit diagnostic `local_solver="generalized_dense"` mode is
requested. Every Symmray two-site writeback records an entry in
`svd_diagnostics`, including the split direction, bond name, `chi`, cutoff,
and the left/right charge sectors kept by Symmray's SVD.
`last_svd_diagnostic` exposes the most recent entry.

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
