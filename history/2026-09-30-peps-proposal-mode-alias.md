# 2026-09-30 — Name proposal-only sampling explicitly

- Scope: user approved `amplitude_mode="proposal"`, retaining `"none"` as
  a compatibility alias.
- Branch / baseline: `develop`, `dacc200`; prior chunk-review edits and
  concurrent tree/backend work preserved. No files staged, committed or pushed.

Added the spelling to `PepsSampler` and `PEPSSampleResult` handling for serial,
grouped, chunked and streamed calls. Both spellings skip amplitude evaluation,
return `ps=None` and q, use equal weights, and retain the probability validation
from the preceding task. Metadata preserves the supplied spelling so existing
`"none"` consumers remain compatible. The default stays `"boundary"`.

Updated the API guide and changelog. Parameterized existing no-amplitude and
draw-equivalence tests over both spellings on NumPy/Torch and all sampling
interfaces; automatic chunks and the exact-Born convergence check exercise
`"proposal"`. Both spellings reject malformed saved probability records.
No numerical contraction policy or dependency changed; the previous task's
upstream audit still applies.

Validation: focused alias/amplitude/chunk and public-API/layout selection:
118 passed, one pre-existing installed-version mismatch (10.32 s).
Smoke: 92 passed, the same mismatch (33.59 s); installed metadata is 0.4.0
versus project 0.5.0. Full `ruff check src tests` and whitespace checks passed.
No numerical suite rerun beyond the affected selection was needed for this
spelling alias; broader numerical results remain in the preceding handoff.
No production simulations launched.
