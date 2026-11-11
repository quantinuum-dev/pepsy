# Reduced full-update factor reuse

Reuse the initial SVD and balance an accepted rank-χ result through a small
core instead of decomposing the reconstructed pair again. This is implemented
in `optimizers/peps/_full_update.py`; no backend registration changes.

For reduced factors A and B, exact QR/LQ factorizations give

$$
{\color{blue}AB}=Q_A\,{\color{purple}(R_A L_B)}\,Q_B,
\qquad R_A L_B=U S V^\dagger.
$$

Absorb √S into both sides. The core is at most χ×χ; the retained rank and
symmetric singular-value gauge match a full-pair SVD up to floating-point
and degenerate-basis choices. Explicit `gauge=True` retains its previous
full final split. Target normalization updates the cached starting factors,
including complex phase. SU starting factors retain their existing path.
Ungauged fitting also reuses the matrix already formed by PSD projection
instead of rebuilding it through a square root and another dense product.

This differs from the earlier downstream prototype, which used only QR/LQ
for final regauging. Small exact-environment tests do not establish unchanged
long-time approximate boundary contractions. The downstream benchmark owns
the matched CPU16 χ6 profile and full-T accuracy comparison.

The subsequent user-directed boundary policy defaults FU to `reuse-direct`
and `n_iter=4`, preserving explicit overrides. This belongs in both constructor
and runtime mode selection. A fitted boundary is a disposable starting guess
for a fresh FIT against the updated target, never a stale accepted contraction.
Seeded Torch SRC is supported by the installed Quimb build; `guess-src` and
`reuse-src` remain explicit controls. This policy can change finite-sweep
approximate results independently of the exact factor-reuse changes above.

One-site FIT cannot increase its initial MPS dimensions. Reusing a small
early cut therefore changed the downstream full-T χ2 cost even with four
sweeps. Dense fixed-rank boundary FIT now opens its warm starting space when

$$
{\color{purple}b_i}<
\min\left({\color{blue}\chi_b},\prod_{j\le i}d_j,\prod_{j>i}d_j\right).
$$

Here b_i is an existing boundary bond dimension and d_j is the product of
outer-index dimensions at boundary site j. Initial conservative cold reseeding
validated the cause but retained costly target compression. The final path uses
public `MPS.expand_bond_dimension(..., rand_strength=0)` followed by
`left_canonicalize_(normalize=False)` on an owned copy. This preserves the old
represented MPS while enabling one-site updates in the requested space.

Expansion is restricted to non-differentiable NumPy/Torch/CuPy arrays; avoid
introducing a rank-deficient QR into differentiated/JAX guesses. Those cases,
and an old MPS exceeding a lowered cap, retain cold seeding. Geometric checks
use metadata and add no device synchronization. Two-site/block FIT grows
locally and keeps its low-rank starts; native Symmray reuse is unchanged.
Tests check the unchanged warm-state arrays against dense contractions, new
bond growth and CPU/CUDA execution. Diagnostics report `reuse-expanded`.

## Native local-solve compatibility

Full-update Quimb ALS now passes public `dense_solve=True`. Its environments
are already dense and native Torch/CuPy arrays cannot enter Quimb's SciPy
LinearOperator branch. This promotes the narrow fix used by the earlier
downstream frozen campaigns; it does not change the shared BP solver's
defaults. `norm_matrix_dimension` reports the reduced environment size.

## Dependency audit, 2026-11-04

Python 3.12; Torch 2.14.1+cu126; Quimb 1.15.1.dev101+g1b64353d9;
Autoray 0.11.1.dev14+g014a3f69a; Cotengra 0.8.3; Cotengrust 0.2.1;
Symmray 0.4.1.dev25+gdc377f033. Inspected installed `Tensor.split`,
`tensor_network_fit_als`, `qr_auto`, native QR/SVD dispatch and the
automatic dense-solve threshold. The public MPS expansion/canonicalization
signatures were also inspected and exercised without adding a new QR kernel.

Reviewed official [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changes](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray](https://github.com/jcmgray/symmray). The requested Symmray
abelian-array documentation URL was unavailable; installed source and the
official repository were used. No environment or installed dependency changed.

- **Adopt:** existing native QR/LQ/SVD and public `set_gates()`/`run()` for
  dependency-checked caches across source barriers.
- **Compatibility shim:** full-update-only dense native ALS dispatch.
- **Defer:** approximate PSD spectra, changing precision/tolerances, removing
  source barriers, and claiming native symmetry/JAX FU support.

Validation and commit status are recorded in the
[session handoff](../../../history/2026-11-04-full-update-factor-reuse.md).
