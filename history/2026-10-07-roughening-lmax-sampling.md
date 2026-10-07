# 2026-10-07 — Investigate roughening longest-path mismatch

- Scope: user's 5×6 `rough_exact_mps` longest-wall disagreement and chi trend.
- Baselines: Pepsy `develop` at `7eb2b94`; examples `main` at `d0cd8bf`.
- Commit status: the user subsequently requested committing and pushing the
  fix. This scoped package commit contains the sampler, regressions, API and
  changelog updates, package evidence, and this handoff. Examples annotations
  remain working-tree edits. The push outcome is reported after verification.
  Preserved pre-existing MPS/JAX changes, documentation, and notebook outputs.

## Findings and changes

Confirmed and fixed float32 inverse-CDF bias in large Torch `VecSampler`
draws. Uniform 25-qubit complex64 sampling returned final-bit one frequency
0.25281 instead of 0.5 before the fix. Float64 CDF accumulation and uniforms,
with full-CDF normalization, pass the actual large-vector regression and a
small deterministic narrow-interval test. Returned probability gradients,
dtype, and device remain preserved. Updated API guide and changelog.
See [package evidence](../docs/development/notes/2026-10-07-vector-sampling-precision.md).

Recomputed five 8192-shot theta_09 datasets: longest-path means and errors
agree exactly with the stored plots. MPS chi=128 to 2048 increases 7.78833
to 8.41284; tree chi=256 to 1024 decreases 8.62781 to 8.55200. Checked
matching physics/order/time settings. Truncation remains considerable, and
these runs span different source revisions.

Both new exact reference sweeps used the affected complex64 sampling path.
Their particular observable bias remains unmeasured. Added a prominent
notebook note and plotting-guide link to the examples investigation at
`experiments/mps_magnetization/benchmark/docs/development/notes/2026-10-07-lmax-sampling-audit.md`.
Preserved all figures and raw datasets; no simulation was launched or stopped.

## New validation

- Sampler, public API, and package layout: 165 passed, two alias warnings.
- Pepsy Ruff and whitespace checks passed.
- Notebook JSON/schema, code compilation, and new local links checked
  separately; the notebook edit is confined to an introductory Markdown note.
  An attempted comparison against an earlier disposable execution found that
  artifact differs from the source's existing outputs, so it was not used
  as a preservation baseline. No outputs were regenerated or edited here.
- Full numerical suite and CUDA validation not run.

## Remaining work

Correcting the exact curves requires new samples, not a notebook refresh.
The inspected archives do not retain final states, so a production rerun
would require the user's launch authorization. The confirmed sampler defect
does not by itself prove the cause or size of the observed lmax gap or the
nonmonotonic tree trend.
