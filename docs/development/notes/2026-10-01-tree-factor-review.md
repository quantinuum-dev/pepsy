# 2026-10-01 — TreeSampler factor follow-up review

Scope: user requested another review, fixes, commit and push. Baseline:
published `develop` commit `45d6b5e`. Reviewed dense cache merging/root reuse,
native canonical movement, sector remapping, prefix normalization, cache
lifetime/accounting, random ordering, device/gradient preservation and the
handwritten API. Unrelated working-tree edits are excluded.

## Findings and changes

- **Adopt: correctness correction.** Dense conditional probabilities can
  have a cumulative endpoint slightly below one. With trailing zero-weight
  physical codes, a uniform draw at `nextafter(1, 0)` then selected a zero
  state and raised the conditional-norm error. Reproduced with both float64
  and float32 source vectors, both strategies and chunked/unchunked calls.
  The new endpoint matrix failed 28 cases and passed four before the fix.
  Dense sampling now divides its CDF by the accumulated endpoint, using one
  as a safe divisor for an invalid zero endpoint. Invalid conditional norms
  still raise; leading and trailing zeros remain impossible selections.
  Source tensor dtype, Born-weight calculation and uniform ordering are
  unchanged. Boundary rounding can change the selected code, as intended.
- **Adopt: documentation correction.** The fermionic API example now selects
  `backend="native"` explicitly. Its prose distinguishes canonical factor
  sampling from the full projected-tree standard reference, and native
  graded source-projection amplitudes from dense compatibility signs.
  The source module labels its array traversal as the dense algorithm.
- **Validated:** native odd global parity on spinless Z2/U1 and spinful
  Z2/U1/U1U1/Z2Z2, with physical and virtual roots. New regressions compare
  seeded samples/probabilities to standard, signed amplitudes to independent
  source projections up to a global phase, and decoded total parity. No
  additional native implementation change was required.

## Environment and validation

Rechecked unchanged installed versions: Quimb `1.15.1.dev66+ge927f06e1`,
Autoray `0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev7+g83fb22865`, NumPy `2.5.2`, Torch `2.6.0+cu124`, cupy-cuda12x
`14.1.1`. Reused the same active task's
[official upstream audit and dispatch evidence](2026-10-01-tree-native-factor-improvements.md#upstream-and-environment-audit),
and inspected NumPy `cumsum`, Quimb `Tensor.isel` and the existing
NumPy/Torch/CuPy cumulative-sum/where dispatch. No dependency upgrade or shim.

Focused endpoint/zero-norm/zero-draw validation after the fix: **48 passed**,
89 deselected in 2.97 seconds. The odd-parity test harness initially assumed
spinless decoded occupations had a spin axis; corrected it to the documented
two-dimensional shape. Probability and relative-sign comparisons already
passed; final validation includes the corrected parity assertion.
Final broad selection: **629 passed, two skipped**, four existing compatibility
warnings in 197.03 seconds. Ran in the activated session environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -ra -o addopts='' \
  tests/test_tree_sampler_validity.py tests/test_tree_factor_sampler.py \
  tests/test_tree_symmray_factor.py tests/test_tree_sampler.py \
  tests/test_tree_canonical_regions.py tests/test_tree_entropy.py \
  tests/test_tree_unitary_stability.py tests/test_public_api.py \
  tests/test_package_layout.py
python -m ruff check src tests
git diff --check
git diff --cached --check
```

Ruff, relative documentation links and whitespace checks pass. Known skips:
CuPy float32 subnormal-source arithmetic and a test requiring two CUDA devices.
Torch CPU/CUDA, CuPy and native Symmray otherwise ran. Logs:
`/tmp/pepsy_tree_factor_review_cdf_before.log`,
`/tmp/pepsy_tree_factor_review_cdf_after.log`,
`/tmp/pepsy_tree_factor_review_validation.log`. Publication is recorded in the
[review handoff](../../../history/2026-10-01-tree-factor-review.md).

**Defer:** production-checkpoint performance, large-bond native and Torch
autograd peak memory. Earlier benchmark tables remain dated measurements of
their respective implementations; this review makes no new timing/memory
claim. Full-package validation is outside this sampling-only review.
