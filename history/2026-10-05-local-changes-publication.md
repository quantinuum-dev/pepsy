# 2026-10-05 — Publish remaining local Pepsy changes

- Scope: user explicitly requested committing all local Pepsy and Gaugy work
  and synchronizing both repositories with their remotes.
- Branch / baseline: `develop` / `7a01b05`; freshly fetched `origin/develop`
  matches the baseline, so no integration merge is needed.
- This record accompanies the commit of all pending source, tests, docs and
  historical handoffs. Push to `origin/develop` follows the local commit;
  remote publication is confirmed by Git, not by this pre-push record.

The snapshot includes JAX SciPy/NLopt solver support, conservative MPO channel
reduction safeguards, sampling/MPI test contract corrections, and reuse of the
shared JAX precision context in tree compression. Existing implementation and
historical records are preserved without behavior edits during publication.

Fresh checks: `python -m ruff check src tests`, AST parsing of changed and new
Python files, and `git diff --check` pass in the shared Python 3.12 environment.
No numerical tests or full suite were rerun for this synchronization. Earlier
scoped validation is recorded in the [solver handoff](2026-10-01-jax-host-solvers.md),
[MPO handoff](2026-10-02-mpo-delinearization-review-fixes.md), and
[sampling handoff](2026-10-01-sampling-test-contract-fixes.md); those results do
not establish full-suite validation of this combined snapshot.

Earlier records saying changes are uncommitted describe their original
sessions; this publication supersedes that status for the files in this commit.
