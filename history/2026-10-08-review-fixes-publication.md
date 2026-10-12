# 2026-10-08 — Review fixes publication

- Scope: user authorized committing and pushing the completed review fixes.
- Branch / baseline: `develop`, `ad8ed05`.
- Commit status: the fixes and evidence below are included in the commit
  containing this handoff. Push success must be checked against the remote.
- Includes PEPS gate conversion, target normalization, scheduling, and
  separate refinement boundary caps; BP residual confirmation, Torch channel
  capture, native fermionic CTMRG, MPS boolean readback, and test isolation.
- Excludes concurrent MPS/JAX tracer-device changes and their documentation
  and tests; those remain in the working tree.

The [full-suite correction record](../docs/development/notes/2026-10-08-full-suite-failure-corrections.md)
records **8,263 passed, 9 skipped, zero failures**. That run included concurrent
MPS/JAX tracing work and preceded the late PEPS cap correction. A subsequent
five-module PEPS check passed **132 tests** on the final PEPS source. These
results describe the tested working tree, not an isolated checkout containing
only this commit. Final PEPS source hashes still match that focused run;
Ruff and whitespace checks passed again before committing.

See the [PEPS review fixes](../docs/development/notes/2026-10-08-full-update-review-corrections.md)
and [cap correction evidence](../docs/development/notes/2026-10-08-full-update-caps-quimb-refinement.md).
Earlier journals' uncommitted status records their original sessions.
Larger-region refinement remains a proposal, with no production integration
or large-D/long-time accuracy certification in this publication.
