# 2026-09-28 — PEPS optimizer and direct sampler review

- Scope: user requested careful implementation review and report for
  `PepsOptimizer` and `PepsSampler`; no numerical fixes requested in this turn.
- Branch / baseline: `develop` / `6486a61`.
- Commit status: review handoff only, uncommitted; existing launch journals
  preserved. No production jobs changed, no source edits or publication.

## Confirmed findings

1. `src/pepsy/optimizers/peps/optimizer.py:944`: `_clean_infidelity` maps
   NaN and arbitrary negative values to zero through `max(0.0, value)`.
   Injecting a NaN boundary result into the public `estimate_infidelity`
   returned 0.0. This can suppress cleanup or make an invalid candidate look
   perfect. The reproduction injects the metric failure; it does not establish
   how frequently a real boundary contraction produces NaN.
2. `optimizer.py:720`: target options retain the run's nonzero cutoff.
   On a 2×2 |0000> product PEPS, applying exp(-0.1i X⊗X) with chi=2,
   cutoff=0.1, cutoff_mode=rsum2 and the default initial/target
   normalization returns bond 1 and records `within_chi`, infidelity=0.0.
   Independent cutoff-zero dense reference gives normalized infidelity
   0.009966711079379076. The returned state has unit norm, so this does not
   depend on disabling normalization. Target truncation is invisible to the
   reported error and conflicts with the exact-target claim.
3. `optimizer.py:878,921`: the normal warm-start path does not forward
   cutoff_mode to compress_all. A run requesting abs sends only max_bond,
   cutoff, and inplace. For random complex128 2×2 D=4 PEPS, seed=2,
   chi=3, cutoff=0.1, this helper retains bonds [2,2,2,2], while explicit
   cutoff_mode=abs retains [3,3,3,3].
4. `src/pepsy/sampling/peps.py:1657,1691`: exact draws contract raw
   conditional magnitudes, unlike exact log_probability, which strips
   exponents. Setting a small PEPS's network exponent to +200 leaves its
   proposal unchanged but sample/sample_batch raise OverflowError; -200
   raises invalid-trace ValueError. Probability queries succeed. Stable
   amplitude contraction alone does not make the entire sampling path
   invariant to extreme overall scale. The API guide already warns that
   arbitrary ill-scaled contractions are not guaranteed, so this is a
   concrete limitation rather than a newly discovered contract violation.
5. `src/pepsy/sampling/peps.py:1916-1923`: grouped samples assign the same
   mutable `group["config"]` list to each shot in the group. A deterministic
   2×2 |0000> PEPS returns three identical object references from
   `sample_batch(3)`; editing the first configuration changes all three while
   leaving their recorded probabilities/amplitudes unchanged. Serial
   `sample(3)` returns independent lists. This is a result ownership defect,
   not a probability-distribution error before caller mutation.

## New validation

- Focused NumPy/Torch selection:
  `python -m pytest -q -ra -o addopts='' tests/test_optimize_peps.py tests/test_peps_sampler.py tests/test_peps_sampler_efficiency.py -m 'not slow and not integration' -k 'not jax'`
  → **245 passed, 98 deselected**, 39 Quimb mode/method FutureWarnings,
  24.12 seconds. CPU library threads capped at 2.
- Earlier broader selection was interrupted during JAX compilation after
  173 passed and 2 skipped. It is not a completed suite result. Large 4×4
  studies, full-suite validation, and a complete JAX/CUDA matrix are not claimed.
- Temporary reproductions and output:
  `/tmp/peps_review_20260928.py`, `/tmp/peps_review_20260928.log`,
  `/tmp/peps_review_focused_20260928.log`.
- Rechecked finding 2 with default normalization and an independent cutoff-zero
  dense target; measured result and reported zero were unchanged. Confirmed
  finding 5 with a deterministic three-shot product-state batch. No second
  broad test sweep was needed because the source and environment were unchanged.
- Imported Pepsy source verified to be this checkout. Installed metadata:
  pepsy 0.4.1, quimb 1.15.1.dev66+ge927f06e1,
  autoray 0.11.1.dev3+g1b476b305, cotengra 0.8.3.dev7+g1d7fd333f,
  symmray 0.4.1.dev7+g83fb22865, torch 2.6.0+cu124.
  Installed tensor_split default cutoff_mode is rel.

## Interpretation

The sampler separates the conditioned ket from future environments and uses
original PEPS amplitudes for importance weights. Approximate boundary samples
still require weights; finite chi can lose proposal support, which weights
and observed ESS cannot repair. Exact amplitude contractions remain a scaling
bottleneck. These are documented limitations, separate from the reproduced
extreme-scale draw failure. Findings above remain unfixed pending follow-up.
