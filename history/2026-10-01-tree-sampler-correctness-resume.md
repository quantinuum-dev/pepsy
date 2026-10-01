# 2026-10-01 — Resume TreeSampler correctness work

- Scope: user requested inspecting and resuming unfinished TreeSampler work.
- Branch / baseline commit: `develop` / `5558857`.
- Commit status: implementation was validated in the working tree; the user
  subsequently requested a local commit (see the follow-up below). Earlier
  vector/factor sampling work is committed in `3bfaaa9`, included in baseline.
  Existing MPI, trajectory and PEPS test changes and their concurrent handoff
  were preserved.

## Implemented

Addressed the remaining September 30 findings: long complex64 conditional
underflow, native single-site flip ratios, detached Torch root-normalization
gradients, invalid/extremely scaled norms, and invalid configuration coercion.
Added scale-safe float64 probability scoring and zero-draw branch handling;
amplitudes keep their existing dtype. Updated API/changelog and clarified the
shallow-array snapshot caveat. The factor strategy remains experimental and
opt-in, with its existing chunk/cache policies.

Detailed changes, dependency audit, independent reference checks and measured
limits are in the [evidence note](../docs/development/notes/2026-10-01-tree-sampler-correctness-resume.md).

## Validation and limits

Final focused selection: **512 passed, two skipped**, four compatibility
warnings, in 96.96 s. Covers new validity/scale/gradient regressions, both
sampler strategies, canonical regions, entropy, unitary stability, public API
and package layout. Ruff, whitespace and local documentation-link checks pass.
Serial before/after 30-site complex128 comparisons preserve every sampled
configuration and probability. Median rescaling overhead: factor **9.4% on
NumPy** (chi=32) and **3.3% on CuPy** (chi=256); standard timings change by
-0.5% and +2.5%, respectively. The corrected CuPy factor path is about 5.87x
faster than corrected standard in that synthetic case. Two repetitions per
variant; these are not production measurements. No allocator peak measured.
The frozen committed baseline fails the new regressions (22 failures in the
selected baseline cases). Full-package checks and production-checkpoint
validation have not been run. The CuPy subnormal-source capability check is
explicitly skipped when upstream float32 arithmetic flushes these values;
normal-component long-tree sampling remains independently tested. The other
skip requires two CUDA devices; one device is available and its Torch/CuPy
paths were exercised. See the linked note for scopes and exact commands.

No commit, publication, default promotion, dependency changes or sibling edits.
All test and benchmark runners have completed.

## Local commit follow-up

The user subsequently requested committing this TreeSampler work. This
handoff is included in the local commit titled `Fix TreeSampler scaling,
gradients and input validation`, together with the implementation, regressions,
API/module documentation, changelog and evidence note. The separate MPI,
trajectory and PEPS test-contract edits and their handoff are excluded.
No push or publication was requested. Existing numerical validation remains
the result recorded above; the commit follow-up repeats Ruff and whitespace
checks without changing numerical code.
