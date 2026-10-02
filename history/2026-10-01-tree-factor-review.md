# 2026-10-01 — TreeSampler follow-up review

- Scope: user requested another review, commit and push of TreeSampler work.
- Branch / baseline: `develop` at published `45d6b5e`.
- Commit status: included in the scoped commit titled `Fix TreeSampler CDF
  endpoints after factor review`; publication to `origin/develop` is checked
  against the remote ref at final handoff. Unrelated solver/MPI/PEPS/trajectory edits and JAX changelog
  entries remain outside this commit.

## Changes and findings

The review reproduced a dense CDF endpoint rounding bug: a uniform very close
to one could select a trailing zero-weight code. Normalize by the accumulated
endpoint, with a safe divisor for invalid zeros. The original conditional-norm
validation remains active. Added deterministic endpoint regressions for both
strategies, chunked/unchunked calls, float32/float64, NumPy, Torch CPU/CUDA and
CuPy. Also corrected the fermionic API example's backend and description of
native versus dense amplitude conventions.

Added odd-parity native regressions with physical/virtual roots and spinless
Z2/U1 or spinful Z2/U1/U1U1/Z2Z2. Seeded samples, Born probabilities, independent
source-projection relative signs and decoded parity are checked. No further
native implementation issue was found. Root/cache changes remain as published.

## Validation and limits

Endpoint/zero-draw/zero-norm checks after the correction: 48 passed, 89
deselected. The pre-fix endpoint matrix failed 28 cases and passed four.
Final broad regression: **629 passed, two skipped**, four existing compatibility
warnings in 197.03 seconds. Covers validity, factor/native sampling, sampler
integration, canonical regions, entropy, unitary stability, public API and
package layout. Ruff, relative documentation links and whitespace checks pass.
Torch CPU/CUDA, CuPy and native Symmray were exercised; skips are the known
CuPy subnormal-source limitation and a check requiring two GPUs. The
[review evidence](../docs/development/notes/2026-10-01-tree-factor-review.md)
records unchanged dependency versions and the reused same-task upstream audit.
Earlier [implementation validation and measurements](../docs/development/notes/2026-10-01-tree-native-factor-improvements.md)
remain historical evidence, rather than this review's final results.
No full-package or new performance/peak-memory validation is claimed.
