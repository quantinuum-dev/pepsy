# 2026-11-11 — Commit pending cluster updates

- Scope: review, validate and commit the user's pending Pepsy changes.
- Branch / baseline: `develop` at `add66d9` before this commit.
- The commit contains the previously uncommitted prepared-cluster geometry,
  PEPO physical-leg reduction and adjacent joint-update changes, their tests,
  API documentation and November 10 handoffs.

## Validation

With the existing Python 3.12 environment and one numerical thread:

- Affected BP/prepared-update and PEPO tests, excluding known simple-update
  warm-start cases: **212 passed, 73 deselected**.
- The same selection before excluding `su_warm_start`: **212 passed**, then
  failed at `test_su_warm_start_preserves_custom_unequal_legs_and_normalization`
  because the installed Quimb `Tensor` lacks `multiply_index_diagonals_`.
  The [earlier investigation](2026-11-10-pepo-physical-leg-reduction.md)
  reproduces this issue with the unchanged warm-start implementation.
- Repository-wide fail-fast: **354 passed, 2 skipped, 1 failed** at
  `test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges`: the
  native exact expectation was zero instead of the dense Jordan–Wigner value.
  This is the previously recorded failure; see the
  [joint-update handoff](2026-11-10-adjacent-joint-cluster.md).
- `python -m ruff check src tests` and `git diff --check` passed.

The full suite is not green; no dependency or unrelated numerical fix was
introduced. The commit is local and was not pushed.
