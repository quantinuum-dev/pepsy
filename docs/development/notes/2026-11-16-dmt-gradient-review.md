# DMT review: degenerate Torch Hamiltonian gradients

**Fixed:** `PauliDMT.evolve` returned NaN parameter gradients for a trainable
`ZZ` bond Hamiltonian, although its gate and observable derivatives are finite.
This was reproduced on CPU with Torch 2.9.1 using both real and complex cores.

## Cause and adopted route

The EIGH-based gate differentiated eigenvectors at repeated eigenvalues.
[Torch documents this limitation](https://docs.pytorch.org/docs/2.9/generated/torch.linalg.eigh.html).
For Torch Hamiltonians requiring gradients, use Autoray's existing
`linalg.expm` dispatch to `torch.matrix_exp`. Other Hamiltonians retain EIGH.
Both routes cache half/full-step transfers within a call and rebuild on the
next call. No spectrum perturbation or global registration is introduced.

**Adopt:** the public native matrix exponential, whose installed Autoray
dispatch was inspected. The installed Quimb `array_split` signature retains
`method`, `absorb`, `max_bond`, `cutoff`, `cutoff_mode`, `renorm` and `info`;
no tensor-factorization route changed. **Defer:** unrelated upstream updates
and native Symmray support.

## Environment and upstream audit

Installed: Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray 0.4.1.dev15+g0374aaa3c.
Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The linked Symmray array documentation remained unavailable. No dependencies
or installed libraries were changed.

## Regression scope

- Dense three-site evolution and gradients for `H = theta * (ZZI + IZZ)`:
  zero/nonzero parameter, real/complex double-precision cores, both conjugation
  directions, shared Hamiltonians and multiple timesteps. The regression
  failed with NaN gradients before the fix.
- Full 3×2 lattice-region checks against each exact gated target during
  genuinely truncated density and traceless-operator evolution, including
  continuation across calls and bounded reports.
- A rejected persistent gate preserves the preceding physical state and
  permits a subsequent update with consistent canonical/trace environments,
  with both default chain protection and explicit lattice regions.

Validation results and remaining limits are recorded in the
[session handoff](../../../history/2026-11-16-dmt-review.md).
