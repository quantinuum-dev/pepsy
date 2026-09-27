# Structural reduction of complete located PEPO traces

> This implementation record predates the composed boundary-factor derivative
> correction. Its measured MPS discrepancies describe that earlier baseline;
> see [cluster optimization status](../cluster_optimization_status.md) for
> the later correction, validation scope, and remaining limits.

## Scope and algebra

The located `PauliPEPOBasis` path used by `PEPOClusterProductExpansion`
stores residual coefficients at each cluster tree's root. Nonroot tensors
are fixed Pauli-history selectors. A complete physical trace kills each
nonidentity Pauli. Induction from the leaves forces every subtree history
to be all-I, whose allocated sector has history index zero. Different trees
have distinct allocated sectors, so their certificates can be unioned with
the trivial sector. The root contributes the identity residual coefficient.

`_build_inhomogeneous_active` records that union on its final active blocks.
Lower-order snapshots used for **operator** subtraction retain all histories.
`to_trace_network()` filters by this static certificate before materializing
site tensors; it never inspects coefficient values or gradient tracking.
Numerical zeros in allowed sectors remain. `trace_nbytes` estimates this
smaller allocation. Full `to_pepo()` is unchanged. Rank-capped SVD trees lack
fixed selectors and disable the certificate. General builders remain on
their existing path; `compact()` conservatively drops the certificate.

This argument applies only to the complete bosonic physical trace, not
partial traces, inserted observables, native symmetry, or arbitrary edits to
the active physical blocks.

## Validation and downstream measurements

- Storage + cluster-expansion suites: **70 passed**. New order-4 2×2 OBC/PBC
  joint noncommuting-factor tests compare full dense PEPO, untrimmed trace,
  and reduced trace, with all Torch coefficient derivatives. Zero and two
  nonzero parameter vectors preserve topology; NumPy agrees. Existing JAX
  derivative coverage passes; rank-capped trees retain all sectors.
- Public API/layout: **58 passed**, 8 deprecation warnings.
- `ruff check src tests` passes.
- Gaugy order-2 4×4 trace storage: 57,600 → 2,304 bytes OBC,
  160,000 → 4,096 bytes PBC. Tested MPS and explicitly Cholesky-conditioned
  CTMRG derivatives now agree with finite differences for circuit and
  single-exponential ITF gauges, including actual chi-1 truncation.
- This is not a universal boundary-gradient fix. On Gaugy's 3×3 order-3 ITF
  fixture, MPS directional discrepancies remain 4.59e-3 OBC / 3.27e-5 PBC.
  CTMRG with reduction `method='cholesky', shift=1e-10` gives 3.82e-10 OBC
  / 1.31e-6 PBC. Default SVD CTMRG also retains a smaller order-2 PBC bias.
  Exact contraction is still the downstream default.
  A subsequent explicit CTMRG shift scan (1e-8, 1e-6, 1e-4) reduced that
  order-3 PBC discrepancy below 9e-10, with costs within 2e-12 of the exact
  contraction. This does not establish a universal conditioning policy.

## Upstream audit and decisions

Same installed environment as the [preceding audit](2026-09-26-pepo-product-trace.md):
Pepsy 0.5.0, Quimb 1.15.1.dev66+ge927f06e1, Cotengra
0.8.3.dev7+g1d7fd333f, Autoray 0.11.1.dev3+g1b476b305, Symmray
0.4.1.dev8+gc45f91457, Torch 2.9.1, JAX 0.8.2. No dependency changes.
Rechecked official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray](https://github.com/jcmgray/symmray); the prior same-task
documentation/source audit also covers the remaining required pages.

Installed `compute_reduced_factor` / `squared_op_to_reduced_factor` SVD
reduction absorbs square roots of singular values. Anomaly detection in the
unreduced downstream trace identifies `SqrtBackward0` at an exact zero;
stabilizing SVD backward does not protect that subsequent square root.
Public Cholesky reduction supports a positive relative shift, but changing
the downstream default regressed a converged small-system derivative test.
That default change was rejected; no global QR/SVD constants were adjusted.

- **Adopt:** algebraically certified trace sectors, existing public opt-in
  Cholesky reduction for the bounded downstream tests.
- **Defer:** a universal boundary-gradient guarantee, native graded trace,
  and whole-wrapper compilation. Forward accuracy does not validate a VJP.
- No compatibility shim, vendored upstream code, or installed-library edit.
