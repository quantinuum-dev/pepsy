# 2026-09-28 — Correctness review of complex-to-real warnings

- Scope: user's approval to investigate and fix the remaining cast warnings.
- Branch / baseline: `develop` / `bc77796`.
- Commit status: working-tree changes; nothing staged, committed, or pushed
  during this pass. The two preceding local commits remain unpushed.

## Changes

- Ordinary Haar MPS/PEPS/TTN constructors reject real storage that previously
  discarded phases and could reduce the norm. Complex seeded results remain
  unchanged; `ps_to_*` is the documented real-product alternative.
- Stabilizer cached constants distinguish zero imaginary storage from
  genuinely complex operators. Single-site Y rotations assemble directly
  in real storage; unsupported complex coefficient operators raise clearly.
- D2 edge-loop norm weights are validated before Quimb's real suppression
  solver. Roundoff residues are accepted, larger imaginary/nonfinite values
  rejected, and observable numerators remain untouched.
- Added focused regressions and updated the API guides and changelog.

See the [cast review](../docs/development/notes/2026-09-28-complex-cast-review.md)
for reproduced values, upstream versions, validation commands/scopes, and
the intentional input-contract changes.

## Validation and limits

- Focused regressions: **14 passed**.
- Closest domain suites: **751 passed, 24 skipped**, with no targeted casts.
- Smoke: **92 passed**. Ruff, focused mypy, and the documented example pass.
- Full suite: **5,175 passed, 129 skipped, 682 warnings in 516.88s**.
  None of the five targeted complex-to-real warnings remains. Other upstream,
  diagnostic, compatibility, worker, and gradient-to-scalar warnings remain
  visible. Local Markdown links and `git diff --check` pass.
- CUDA/CuPy coverage remains unavailable. The existing Quimb scalar solver
  is not certified differentiable; its Torch gradient-to-scalar warning is
  separate from the five resolved casts.

No dependencies, source modules, benchmark folders, or documentation tools
were added. Remaining VMC fallback diagnostics, larger refactors, upstream
NumPy deprecations, and publication are outside this focused pass.
