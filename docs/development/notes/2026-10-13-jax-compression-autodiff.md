# JAX compression and differentiation — 2026-10-13

The NetKet boundary adapter now honors Quimb's explicit compression method.
Fixed sweeps are differentiated through their actual updates. An implicit
fixed-point derivative is a separate algorithm requiring convergence and a
gauge prescription; it is not implemented by these changes.

## Implemented and tested scope

| Algorithm | JAX differentiation | Main control |
| --- | --- | --- |
| Quimb zip-up | Unrolled SVD compression | Fixed `chi`, `cutoff=0` |
| Quimb SRC | Differentiation with a fixed sketch | Explicit integer `seed` |
| Quimb SDC | Unrolled deterministic compression | Fixed `chi`, `cutoff=0` |
| Quimb density matrix (`dm`) | Native eigendecomposition derivative | Resolved-spectrum checks only |
| Quimb one-/two-site FIT | Unrolled finite sweeps, including the guess | `max_iterations`, `tol=0`, `bsz=1/2` |
| Pepsy `FIT.run_eff` | Unrolled one-/two-site sweeps; checkpoint compatible | `n_iter`, `rtol=None`, `collect_split_diagnostics=False` |

`contraction_opts={"method": "dmrg"}` and `{"method": "dmrg2"}` are
boundary aliases of Quimb's `fit`, with block sizes one and two and two
directional sweeps by default. They do not dispatch to Pepsy `FIT` or reproduce
`MpsOptimizer`'s block warm-up schedules. Ordinary `fit` retains a ten-sweep
budget. The deterministic default guess is `zipup`; an explicit `tn_fit`
overrides it. Existing legacy `mode` selections remain accepted.

One-site fitting uses QR for its fixed-rank gauge moves. Its initial
compression can still use SVD. Two-site fitting splits updated blocks with
SVD. Iteration count, retained rank, initialization and convergence are
independent controls. A fixed chi alone does not make adaptive stopping
traceable. SRC sketches are held fixed when differentiating; NumPy and JAX
RNG streams differ even with the same integer seed.

## SVD correction

The forward thin SVD is unchanged. Its reverse rule keeps the complete thin
spectrum and restores zero cotangents on discarded columns. Resolved
reciprocals are exact; unresolved gaps use the compact scale-relative
stabilizer already used by Torch. Projectors onto a fully retained repeated
subspace have finite-difference checks when separated from the discarded
spectrum. Individual singular vectors have no unique derivative there,
and a degeneracy across the rank cut remains nonsmooth. General paired
reconstruction derivatives at retained degeneracy are not guaranteed by a
generic factor-wise rule: review found that a symmetric correction in `U`
would repair reconstruction but incorrectly change a subspace projector.
That attempted extension was removed; composed-factor differentiation needs
its own forward map and gauge-invariant derivative contract.

The rule transposes a real-linear differential with public JAX APIs, preserving
complex cotangent conventions and rectangular complement terms. It reuses
the saved decomposition instead of computing another SVD during backward.
Directional checks include resolved reconstruction and retained-degeneracy
projectors, real/complex rectangular matrices, zero singular values and
scales `1e-30` through `1e30`.
These checks do not certify every gauge-dependent factor observable or higher
derivatives. The custom VJP does not add forward-mode support.

## Literature and decision

[Liao et al., PRX 9, 031041 (2019)](https://arxiv.org/abs/1903.09650)
describe spectral-gradient stabilization, checkpointing and differentiation of
converged tensor-network environments. A stabilized derivative can be a
surrogate near unresolved spectra; finite values alone are insufficient
accuracy evidence.

[Francuz, Schuch and Vanhecke, PRR 7, 013237 (2025)](https://arxiv.org/abs/2311.11894)
identify discarded-spectrum corrections and the gauge issue in implicit CTM
differentiation. Keeping the full thin spectrum avoids the approximation of
treating nonzero discarded singular values as zero. Their CTM-specific gauge
solution is not automatically an MPS-FIT implementation.

[Successive randomized compression (2025)](https://arxiv.org/abs/2504.06475)
motivates SRC as a low-cost MPO–MPS compressor. Its performance and error
comparisons do not certify derivatives of a particular random realization.

**Adopt:** explicit Quimb methods, deterministic/seeded initialization,
fixed sweep budgets, stabilized full-spectrum reverse derivatives and
checkpointed Pepsy FIT. No dependency upgrades or installed-source changes.
**Defer:** implicit FIT gradients, adaptive-rank differentiation, and a
degeneracy-safe density-matrix projector rule until independently validated.

## A smarter converged solver

For a full alternating sweep map `y = f(y, theta)` on real tensor coordinates,
the adjoint equation is

$$
\color{blue}{(I-\partial_y f)^T\lambda = \partial_y L},\qquad
\partial_\theta L_{\rm total}
=\partial_\theta L+(\partial_\theta f)^T\lambda.
$$

A mature implementation needs a fixed rank and schedule, a measured fixed-point
residual, an explicit canonical gauge or projection of gauge-null directions,
and a matrix-free adjoint solve with a checked residual. Validate against long
unrolling and finite differences, including complex and repeated-spectrum
cases. A failure must report nonconvergence rather than supply an unchecked
gradient. A few unconverged sweeps must retain the finite-iteration derivative.

For the current small sweep budgets, wrap a pure function that reconstructs
`FIT` from its input arrays and returns output arrays in `jax.checkpoint`.
Construct the mutable Python fitter inside that function. This recomputes
updates in backward while preserving the finite-iteration derivative; it does
not assert a fixed point or detach the initial guess. Fresh regressions cover
both checkpointed and ordinary one-/two-site `run_eff` on dense complex MPS.

## Environment and limits

Installed and inspected: JAX/JAXlib 0.11.2, Torch 2.14.1+cu126, NetKet 3.22.4,
Quimb 1.15.1.dev91+gb3bf45a50, Autoray 0.11.1.dev14+g014a3f69a,
Cotengra 0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev18+g44fc5ae53.
Quimb's boundary `method` and callable compressor dispatch, FIT's `bsz`/
`max_iterations`/`tol`, SRC's seed handling and JAX's SVD differential were
inspected. Official Quimb/Cotengra changelogs and Autoray/Symmray source were
checked; the Symmray array-documentation web reader was unavailable.

Additional draft-reported CPU decomposition checks: **52 passed, 63 deselected**, including
native-oracle gradients on resolved spectra, retained degeneracy,
rectangular reconstruction, scale extremes and adaptive QR. Final combined
results are recorded in the [handoff](../../../history/2026-10-13-jax-compression-autodiff.md).

Additional draft-reported RTX A5000/JAX CUDA boundary checks: **4 passed, 18 deselected**,
241.55 s including compilation. Three checks exercise genuinely truncated
3x3/D=4 native fermionic zip-up, one-site and two-site fitting with
eager/JIT value agreement and complex directional finite differences.
The fourth uses a dense spin PEPS to run native NetKet sample-space SR and a
real-coordinate TDVP step with one-site fitted boundary amplitudes, retaining
PEPS shapes and D. It does not certify a full fermionic TDVP chart.
These are correctness checks, not a GPU speedup or physical convergence claim.
The draft-reported selections were not independently rerun during the final
staged-tree validation. This handoff's verified selections use CPU JAX; their
commands and results are in the linked history entry.

Native flat Z2 deterministic compression has separate checks from dense
compression. Sparse charge storage and randomized native Symmray compression
retain their upstream capability limits. Torch's existing compiled boundary
derivative routing remains in place; the zero-log cleanup is inference tested.
