# Original-paper DMT review

- Scope: review original density DMT behavior and API against
  [White et al.](https://arxiv.org/pdf/1707.01506), and make focused fixes.
- Baseline: `develop` at `56c7365f`. Existing sampling edits, test-profile
  changes, and unrelated history remain untouched; nothing committed/pushed.
- Documented original-paper defaults, Pauli versus ordinary cutoff defaults,
  full-region check controls, bond-local versus full-MPO refinement, and the
  fact that a single Pauli gate caps only its acted-on bond.
- Clarified that the relaxed fallback uses zero-cutoff SVD seeds rather than
  the paper construction's free-residual cutoff. This is existing behavior,
  not a change to the fallback's numerical policy.
- Fixed an extreme relaxed-weight dtype overflow by rejecting weights beyond
  the real component dtype's representable metric range before mutation.
  Added complex64 and complex128 transactional regressions, plus an explicit
  check of the persistent adjacent-gate cap behavior.
- [Evidence and upstream audit](../docs/development/notes/2026-11-17-original-dmt-review.md).

## Validation

- Focused constraint suite: 29 passed, one skipped (CuPy unavailable), before
  the additional acted-on-bond assertion. The DMT domain selection passed
  173 tests and skipped one CuPy case. The two extreme-weight regressions and
  acted-on-bond assertion were rerun separately: three passed.
- Full-suite attempt: 104 passed, seven skipped, stopped at the previously
  documented BP Autoray namespace collision. No DMT code runs in that failure.
- Ruff and `git diff --check` passed during review. No new speedup or
  long-time accuracy claim is made; both require targeted benchmarking.
- Subsequent user-requested publication also includes the
  [PEPO DMT design discussion](2026-11-17-pepo-dmt-discussion.md).
