# 2026-10-30 — saved-factor QR backward for dense PEPO updates

- Scope: investigate Gaugy PEPO runtime, test CTMRG, and assess Torch/JAX compilation.
- Branch / baseline: `develop`, `e658bed6b0dc1c472f38a4afb4cdfe783873a949`.
- Commit status: local, uncommitted changes; no push. Pre-existing MPS/sampling
  edits and their changelog entry are preserved.

## Implemented

[Torch QR](../src/pepsy/backends/linalg_torch.py) now computes the nonsingular
first-order VJP from saved Q/R. Adaptive eager backward avoids a second QR and
nested autograd call; the common single-matrix path also skips boolean-index
gathers. Tall, square, wide, batched, real and complex factors are covered.
The returned complex derivative is materialized, preserving direct NumPy export.
Zero pivots/nonfinite VJPs retain the previous guarded regularized path. The
separate FX capture path and native simulation default are unchanged. No
precision, cutoff, normalization, public configuration or SVD policy changed.

The [package guide](../docs/api/package.md#backend-compatibility) documents the
scope. Batched SU and whole-PEPO compilation are proposals, not implemented APIs.

## Validation

- Focused CPU QR/adaptive/capture/backend selection: **112 passed, 89 deselected**.
- Final CUDA [saved-factor tests](../tests/test_qr_saved_factors.py):
  **62 passed, 62 deselected**. Compare against independent native Torch
  differentiation, including separate Q/R cotangents and both complex precisions;
  public-policy tests reject any backward QR refactorization.
- Gaugy `test_pepog_gradients.py`, `test_pepog_conventions.py`,
  `test_pepog_z2.py`: **79 passed, 3 failed**. All three failures are
  `test_native_zero_overlap_su_gradient[..., True]`, for `pepog_slice_loss`,
  `loss_pepo_color`, and `loss_pepog`. They fail physical-Z2 sector validation
  before QR. The same three failures were reproduced using the unchanged
  baseline linear-algebra module in an isolated process. They remain unresolved.
- `python -m ruff check src tests`: passed. Full package suite not run.
- Complete GPU PEPO objective comparisons retain the same saved target,
  `complex128`, bond 4, boundary MPS 8, cutoff `1e-12`/`rsum2`, all ten slices
  and 243 gradient coordinates. Original/new/new/original ordering separates
  warm timing from setup. Directional finite differences provide an additional
  derivative check; no optimizer trajectory was rerun.

## Evidence and limits

The [downstream record](../../gaugy_examples/research/diagnostics/2026-10-30-pepo-hv-circuit-gpu/README.md)
owns timings, raw gradients, scripts, source hashes, CTMRG checks, and literature.
Its `qr-reuse/` measures the initial saved-factor implementation; `qr-reuse-fast/`
measures this final single-matrix path. Baseline source SHA-256 is
`07ae712891d2d8196364a164043acc6df4b1e8ff81f30f7102f41926b69c6ce9`.
Baseline-failure and final downstream logs are retained in `qr-validation/`.

For the complete ten-slice loss/gradient at the saved final state, the two warm
baseline calls take 35.502/35.448 s and the new calls 31.040/31.239 s:
median **35.475 → 31.140 s**, 12.2% less time (1.139× speed). The loss is
identical and the maximum difference across 243 gradient coordinates is
`7.25e-16`. Final-slice finite-difference errors are `5.50e-9`/`1.25e-8`
at steps `1e-5`/`1e-6`. These are fixed-state evaluations, not a new fit.

CTMRG 8/16 gives finite, directionally checked gradients at both saved states
on the same 4×4 torus, using `quimb_ctmrg_projector_compat`. CTMRG 4 is more
finite-difference-step-sensitive at the initial state. Replacing the trace
contractor gives no material speed benefit in the measured interior slice:
gate updates dominate. The original training already had both periodic seams.

The larger performance opportunity is prepared tensor-array SU execution with
batched independent updates/slices and compiled forward/backward kernels.
Dynamic truncation and singular-chart semantics must be preserved; switching
backend or fixing every bond at rank four is not an equivalent optimization.

The isolated common QR VJP compiles under Inductor `reduce-overhead`, with
matching output and a small warm change from 159.3 to 154.5 µs per call;
Inductor warns about unsupported complex code generation. Whole public PEPO
loss capture stops first at a dynamic capability import. Neither probe
implements or benchmarks a compiled complete PEPO executor.
