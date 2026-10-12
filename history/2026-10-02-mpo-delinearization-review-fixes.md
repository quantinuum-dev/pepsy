# 2026-10-02 — Fix the three MPO review regressions

- Scope: user-authorized fixes for the three findings in the
  [latest-ten-commits review](2026-10-02-ten-latest-commits-review.md).
- Branch / baseline commit: `develop` / `d7d509b`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.

## What changed

- Dense structural reductions check individual channel reconstruction and
  opposing-tensor scales before replacing a bond. Applies to direct automaton
  arrays, converted MPO tensors and existing structural cleanup callers.
- Trainable Torch tensors retain channels, preserving parameter derivatives at
  zero; the existing JAX tracing fallback also remains conservative.
- Unsupported decomposition dtypes retain their original tensors, fixing
  Torch float16 CPU construction. Skipped passes record their reason.
- Added 28 independent regression cases, updated one gradient test's reduction
  expectation, and documented the fallback and local error safeguards in the
  operator API guides and changelog.

## Validation and limits

- Relevant operator/public-API selection: **244 passed, 5 warnings**.
- New non-JAX/non-CuPy cases on original HEAD: **20 failed, 8 deselected**.
- Original three standalone reproducers pass; Ruff and `git diff --check` pass.
- No full-package suite or performance benchmark was run for this fix.
- Conservative safeguards can retain larger bonds. They do not certify a
  global operator error bound; trainable channels are not numerically reduced.

See the [dated numerical and upstream audit](../docs/development/notes/2026-10-02-mpo-delinearization-review-fixes.md)
for implementation rationale, dependencies, commands and validation scope.
The earlier review remains historical evidence. Existing unrelated solver,
sampling and test changes were preserved. No Gaugy implementation changes were
needed, and no unresolved blocker remains for the three reported issues.
