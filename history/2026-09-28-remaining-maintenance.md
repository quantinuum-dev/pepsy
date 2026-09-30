# 2026-09-28 — Remaining maintenance and publication

- Scope: finish the requested cast review, VMC diagnostics, four targeted
  readability changes, warning/hardware investigation, then push and check CI.
- Branch / implementation baseline: `develop` / `6750b19`.
- Commit status: this handoff accompanies the validated implementation batch;
  publication is the next authorized action after creating the commit.

## Changes and evidence

The cast batch is committed as `6750b19`. The two preceding reviewed commits
are `5ae50de` and `bc77796`. The new batch separates boundary dispatch,
BP baseline contraction, VMC result assembly, and PEPO residual fitting, and
adds bounded fallback diagnostics. See the
[detailed evidence](../docs/development/notes/2026-09-28-remaining-maintenance.md)
and [VMC API](../docs/api/vmc.md#connected-amplitude-contract).

## Validation and publication

Focused numerical checks and real/complex baseline gradient comparisons pass.
Final full suite: **5,185 passed, 129 skipped, 685 warnings in 512.58s**.
Default smoke: **92 passed, 2 warnings in 18.47s**. Ruff, the two focused CI
mypy targets, 28 local Markdown link targets, and diff whitespace checks pass.
Scoped Pylint design checks remain advisory and report large functions.

No complex-to-real warnings remain. Quimb shape-assignment warnings account
for 605 occurrences; one Loky worker-stop notice recurred in the passing
NumPy exact-batch phase-pass test. The warning source is recorded in the
detailed evidence; its timeout/memory cause is not established.

At this commit's preparation, the remote `develop` baseline is `f410fa0`.
SSH fetch succeeds. The CLI's token is invalid, but the connected GitHub tool
can read workflow runs, so it can check the requested push without changing
authentication. Hosted results are available in
[the CI workflow](https://github.com/quantinuum-dev/pepsy/actions/workflows/ci.yml).
This local-validation record precedes publication and does not claim a hosted
result; the session's final report records the pushed revision and CI outcome.

## Unresolved external limits

Quimb shape-assignment warnings reproduce outside Pepsy. Worker-stop notices
require monitoring; their root cause is not established. CUDA/CuPy remain
unavailable, and this Torch build lacks native Metal QR. No new dependency,
Sphinx builder, benchmark directory, or warning suppression was added.
