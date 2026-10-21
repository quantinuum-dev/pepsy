# QR derivatives and periodic direct compression — 2026-10-21

Baseline: Pepsy `2aacd87`, downstream Gaugy `2c97228`. This follows the
Gaugy GPU review and repairs dependency causes of five of its seven failures.
The other two were Gaugy loss-derivative cancellation. See the
[handoff](../../../history/2026-10-21-qr-periodic-boundary-fixes.md) for validation.

## QR policy

The adaptive JAX QR pullback selected regularization whenever a diagonal
pivot was below `eps * max(abs(a))`. This changed valid derivatives at small
nonzero pivots and broke cancellations in composed MPS costs. For
`a = diag(1, 1e-8)` and linear weights `[[.2, .3], [.4, .5]]`, the derivative
of `sum((q @ r) * weights)` should equal the weights. The old policy returned
approximately `[[.2, 2.9997e-5], [.4, 4.9995e-5]]`; the corrected policy has
zero error in this check.

Only an exactly zero primal QR pivot selects the regularized extension.
An additional CPU right-overlap check exposed a second problem at exactly zero
pivots: the tall/square fallback also regularized the directly known `Q dR`
contribution. Writing the full-rank rule as that direct term plus a residual
solve leaves only the residual regularized. For a square `diag(1, 0)` linear
reconstruction example, the old extension erased the second gradient column;
the corrected extension returns the analytic weights. This also reduces the
downstream CPU right-overlap finite-difference error from `2.26e-2` to about
`2e-9`. The wide-matrix rule already retained its direct `Q dR` term.

The branch remains independent of incoming cotangents, preserving linearity
and transposition for implicit FIT. No forward factorization, rank cutoff,
or target is changed. Exactly singular charts retain a surrogate extension;
this does not promise differentiability at rank crossings or protect every
possible overflow at extremely small nonzero pivots.

## Upstream audit

Installed versions: Quimb `1.15.1.dev96+g5656a9bb2`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Symmray
`0.4.1.dev20+ge8c907157`, Torch `2.14.1+cu126`, JAX `0.11.2`.

Consulted the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable to the browsing tool; installed source and official repository
were used instead. No installed dependency was edited.

Classification: **adopt** the public callable compression API. Installed
`tensor_network_ag_compress` and `tensor_network_1d_compress` accept callable
methods. The cyclic 2D boundary sweep uses the arbitrary-geometry dispatcher,
whose named method table lacks `direct`. Exported
`tensor_network_1d_compress_direct` handles long-range bonds during its
canonicalization pass before the SVD sweep. Passing this actual compressor
fixes dispatch while retaining cap/cutoff and backend behavior. Pepsy checks
both dispatch capabilities before use and raises if unavailable. No algorithm
substitution, copied upstream implementation, or process-global patch is used.

## Regression design

The full run also reproduced a pre-existing NetKet eager-normalization bug
against untouched `2aacd87`. Flat NumPy Symmray blocks were divided by a JAX
norm scalar, converting their numeric blocks while retaining NumPy sector
metadata. Physical selection then incorrectly dispatched NumPy `take` on JAX
blocks. **Adopt:** compute the block norm and logarithmic scale with Autoray's
native dispatch. Both NumPy inputs and JAX replacements retain their backend;
JIT/autodiff, physical selection and fermionic signs remain on their existing
paths. The original flat-Z2 test passes unchanged after this correction; the
same cause affects the bosonic small-norm TDVP case. No Symmray code is patched.

- QR reconstruction and analytic composed derivatives: real/complex, tall,
  wide and square matrices, nonzero pivots `1e-8` and `1e-16` on CPU/GPU,
  plus exactly zero pivots with complete square Q bases; retain existing
  singular, batched, and transposable-pullback tests.
- Periodic direct boundaries: NumPy/Torch/JAX, both sweep axes, two parameter
  states, exact high-cap contraction and actual rank-one truncation. Compare
  live derivatives to finite differences and retain source arrays.
- Native Z2 bosonic and fermionic cylinders: exact-limit norm comparison and
  preservation of native symmetry arrays. These are distinct from dense
  derivative tests and do not certify all native truncated gradients.
- Unsupported callable dispatch raises explicitly; downstream Gaugy retains
  the original failing boundary and MPS tests unchanged.
