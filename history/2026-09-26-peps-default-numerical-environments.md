# 2026-09-26 — Make FIT-style numerical row environments the default

- Scope: user clarified that FIT caching means actual contracted environments,
  not just Cotengra path caching, and explicitly requested this as the default
  with auto-hq local contractions after checking correctness.
- Branch / baseline: `develop`, `a13031b`.
- Commit status: uncommitted. Preserved earlier changes; no environment installs,
  commits, publication, or production-job changes.

## Change

Boundary PepsSampler now defaults to `row_cache_mode="factored"`, a 64 MiB
estimated row-cache budget, and independent `row_contraction_opt="auto-hq"`.
This enables the already-implemented numerical right-suffix/left-prefix reuse.
Full-network reusable planning remains separate. Explicit zero-budget reference
and dense-transfer modes remain available; estimates above budget fall back to
reference. Exact-mode behavior and all truncation/repair policies are unchanged.

The clarified default supersedes the opt-in decision in the prior planner-policy
handoff. It is not a promise that factored caching always wins on timing or that
64 MiB bounds total memory. The benchmark default and active API/changelog
wording now agree. The roughening integration already selected this policy.

Kept tests that deliberately inspect dense transfers or reference behavior on
those explicit routes. Added a default-path regression that prohibits full-row
local-rho contraction, checks actual first-row cache reuse and separate later-row
prefix environments, compares all small-state Born probabilities, verifies
source arrays, and checks refresh invalidation. Both NumPy/Torch and both
Quimb-MPS/DMRG future engines are covered.

## Validation

- Default numerical-environment proof: 4 passed, 46 deselected.
- Expanded dense/factored rare-prefix regression: 12 passed, 230 deselected,
  covering NumPy/Torch/JAX and complex64/complex128.
- Smoke profile: 162 passed, 1 skipped, the same known version mismatch as
  API/layout. No new smoke failure.
- Full direct sampler/feature run: 280 passed; six failures were the old
  rare-prefix tests asserting mode "transfer" while the correct default mode
  is now "factored". Their numerical likelihood assertions passed before the
  mode assertion. The already-running process had imported those old tests.
  Updated coverage explicitly exercises both representations: all 12 passed
  in the targeted rerun. Together, 292 final direct-sampler cases are covered;
  no production-code change was needed for those six test expectations.
- Roughening PEPS integration suite: 26 passed on current defaults.
- Public API/layout: 58 passed, one known installed-version mismatch:
  runtime/distribution 0.4.1 versus checkout 0.5.0. No assertion/environment edits.
- Ruff (src/tests/benchmark) and git diff --check passed.
- An initial test collection caught misplaced keyword arguments while making
  legacy test routes explicit; corrected before numerical validation.
- No new isolated GPU or full repository numerical suite was run.

The patch-tool sandbox startup failure persists; tracked edits used asserted
replacements. No approval rejection occurred.
