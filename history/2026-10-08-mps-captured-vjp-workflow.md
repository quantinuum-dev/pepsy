# Captured MPS first derivatives for Gaugy evaluation/optimization

Follow-up to [MPS autodiff compilation](2026-10-08-mps-autodiff-compilation.md),
authorized by the user as part of the Gaugy/Pepsy MPS workflow work.

Implemented a fixed-shape FX-capture path for adaptive Torch QR backward.
Undefined native QR inputs are replaced with full-rank rectangular identities
before computing the native candidate; masks select the regularized extension
only for active zero-pivot/nonfinite blocks. Captured runtime tensor assertions
retain nonfinite-gradient and deferred zero-norm checks. Eager QR still uses
its existing conditional implementation and warnings. Native/adaptive rank
policies avoid a redundant host scalar read when no warning/error is requested.

Dense two-site Torch/JAX zero-cutoff replay now uses an exact fixed operator
bond for native `dm` and `zipup`, as already done for direct. Output state
compression still uses the requested method and cap. This avoids a hidden
operator-SVD cutoff before the differentiable state compressor. No installed
upstream code, native Symmray route, or DMRG/FIT algorithm was changed.

Gaugy captures complete forward/VJP array programs through these public replay
and linalg APIs, with a first-order autograd wrapper. Forward-only FX capture
would discard custom SVD/QR backward semantics. Torch `aot_eager` full graphs
pass; Inductor still fails complex layout handling. Local direct JAX gradients
remain invalid on the gauge fixture, while dm/zipup pass. An experimental JAX
SVD-VJP replacement did not resolve that and was removed.

Validation in the activated shared Python 3.12 CPU environment:

- `test_torch_qr_capture`, `test_qr_adaptive`, `test_jax_qr_adaptive`,
  `test_torch_svd_gradients`: **61 passed**, including changed full-rank,
  rank-deficient, zero, wide, tall and batched inputs through the same captured
  QR program.
- `test_mps_compression_modes`, `test_mps_normalization`, `test_backends`:
  **234 passed**.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Integrated tests and benchmark evidence are in the
  [Gaugy workflow record](../../gaugy/history/2026-10-08-mps-eager-overlap-workflow.md).

The dependency audit from the active preceding task was reused; versions were
unchanged. Classification: narrow FX compatibility support plus adoption of
already-public Quimb modes. No dependency upgrades or GPU validation.

Changes remain local and uncommitted/unpublished. Shared PEPS, BP and other
concurrent edits were preserved. A Gaugy flat-boundary test found an unrelated
upstream arbitrary-geometry `method='direct'` incompatibility; no full-suite
success is claimed.
The same failure reproduces using clean archived HEAD source from both packages.
