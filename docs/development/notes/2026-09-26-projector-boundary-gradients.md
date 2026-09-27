# Composed boundary-factor derivatives — 2026-09-26

Published correction: `cf1d84c`, included in `develop` merge `b4c4631`.
See the [current status ledger](../cluster_optimization_status.md) for the
downstream API, merge validation and remaining limits. Measurements below
retain their original scope.

The failing Gaugy order-3 OBC contraction contained a 35×35 QR input of
numerical rank 7, with nine exactly zero QR pivots. Native QR backward is
undefined there; a regularized derivative was finite but biased. Increasing
chi did not fix it. The final closed-network observable can be smooth even
though that intermediate choice of factors is singular.

## Implementation and mathematical contract

`contract_flat(..., method="mps", mps_factorization="projector")` installs
the package-owned `pepsy:projector` method through Quimb's public split-driver
registration. It is used for canonicalization, both reductions, and actual
compression. Existing drivers and Pepsy's default `"qr"` are unchanged.

For a thin SVD `A = U diag(s) Vh`, apply Quimb's requested cutoff/max-bond
policy and then remove numerical nulls below `max(m,n)*eps*s[0]`. With the
retained rank r, return `Q=U_r`, `B=diag(s_r) Vh_r`. Left absorption uses the
transposed construction, including for complex arrays. Retained numerical
rank is a local chart choice, not symbolic removal of allowed histories.

A closed-network loss is invariant under `(Q,B) -> (QW,W†B)` for unitary W.
Choose parallel transport `Q†dQ=0`. If `Ud,sd,Vhd` denote discarded thin-SVD
directions and `gQ,gB` are cotangents, define elementwise

```text
rho = sd[:,None] / sr[None,:]
gap = (1-rho)*(1+rho)
H = Ud† @ gQ + sd[:,None] * (Vhd @ gB†)
K = (H / sr[None,:]) / gap
gA = Q @ gB + (Ud @ K) @ Vhr + Q @ (K*rho)† @ Vhd
```

For a tall matrix, add the thin-SVD left-null complement
`((gQ-U@(U†@gQ))/sr) @ Vhr`. Evaluating H spectrally avoids subtracting
large `A@gB†` components before division by small retained singular values.
Only retained/discarded gaps occur. Degeneracy entirely within the retained
subspace is harmless for the paired observable; differentiating individual
singular vectors would incorrectly introduce internal singular denominators.

This is the composed derivative on a locally fixed-rank chart with a
resolved boundary gap, not a regularized raw QR/SVD VJP. The implementation:

- checks that `Q†gQ+B gB†` is Hermitian within matrix-product roundoff,
  rejecting non-gauge-invariant singular-vector observables;
- rejects an unresolved retained/discarded gap;
- handles an all-zero input with zero incoming cotangents by returning zero,
  and rejects nonzero cotangents at that undefined subspace chart;
- supports dense real/complex NumPy forward and Torch first derivatives;
  uses the scoped Torch policy's forward SVD driver/device/fallback;
- rejects unsupported renormalization and absorption conventions. JAX,
  native Symmray, batched matrices, and higher derivatives are not supported.

Rank growth into a pruned null-null block need not have the derivative of
this fixed-rank chart. Neither a stable forward value nor passing selected
directions proves differentiability across rank/cutoff changes. Check complete
parameterized losses; do not use this primitive for arbitrary factor losses.

## Evidence

`tests/test_projector_split.py` checks real/complex, tall/wide/square matrices,
left/right absorption, all entry gradients against independent native SVD
on resolved spectra, constant-rank factor perturbations, repeated retained
singular values with gradcheck, zero inputs, rejection contracts, all six
cutoff modes, forward-policy restoration, input ownership, and full-network
value/gradient agreement against exact contraction including exponent scale.

Gaugy's complete public loss regressions cover 3×3 order-3 OBC/PBC at chi 1
and 16, perturbed parameters, two directions and two finite-difference steps,
and 4×4 OBC t=1/depth=10 at chi 1, 4, and 16. Chi 1 changes the cost and
therefore exercises real truncation. At chi 16 all parameter gradients also
agree with the exact contraction of the same cluster PEPO.

Measured prototype OBC directional errors: 3×3 chi16 `3.46e-10`, chi1
`2.52e-10`; PBC chi16 `9.53e-10`. Production 4×4 identity-gauge error was
`4.42e-10`, replacing an old error of about `6.63e8`. Full order-3 sweeps at
chi4/16 each took about 38 s, lowering exact-cluster-validated mean local
cost `2.0610752e-5 -> 1.2846710e-5`; final directional error was below
`1.71e-10`. These are derivatives and optimization of a finite-cluster
approximation, not global circuit fidelity or a universal chi convergence claim.

## Upstream boundary

Audited installed Quimb 1.15.1.dev66+ge927f06e1, Cotengra
0.8.3.dev7+g1d7fd333f, Autoray 0.11.1.dev3+g1b476b305, Symmray
0.4.1.dev8+gc45f91457, Torch 2.9.1, JAX 0.8.2; Pepsy baseline `1d327e6`.
Read installed `tensor_canonize_bond`, `tensor_compress_bond`, boundary core,
split registration, and cutoff trimming. The public registration is adopted;
one narrow helper uses Quimb's private trim API, already used by Pepsy's raw
split adapter. Keep its compatibility check and six-cutoff regression tests.
No upstream source was edited or vendored. No Cotengra path/slicing behavior,
Autoray global derivative policy, or Symmray graded handling was changed.
Cotengra's main documentation was also checked. The requested Symmray
`abelian_arrays.html` page was unavailable; the official repository and
installed dispatch implementation were used instead.

References checked: [Torch QR rank requirement](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html),
[Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra changes](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray), and
[Symmray source](https://github.com/jcmgray/symmray).
Keep cutoff explicit across Quimb default changes. Native symmetry and
higher derivatives are deferred rather than emulated by densification.
