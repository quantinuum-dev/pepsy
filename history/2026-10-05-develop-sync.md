# 2026-10-05 — Sync develop while preserving local work

- Scope: pull the latest Pepsy commits and synchronize the current branch.
- Branch / baseline: `develop` / `3f316df`; fast-forwarded to `bec773a`,
  matching the freshly fetched `origin/develop` (two incoming commits).
- Commit status: no new commit or push. Existing local work remains unstaged;
  this handoff is untracked.

Saved tracked and untracked work in the stash named
`pepsy pre-sync local work 2026-10-05`, then restored it after the update.
The stash is retained as a backup. Resolved the sole conflict in
`CHANGELOG.md` by retaining both upstream and local additions. Local edits
to `tests/test_mpi.py` and `tests/test_trajectory_noise.py` already matched
upstream and therefore no longer appear in the working-tree diff.

Validation: SHA-256 checks confirm all 13 other previously modified files
and all eight previously untracked files are unchanged. Reviewed the
changelog diff; `git diff --check` passes, the index is empty, there are no
unmerged entries, and ahead/behind counts are `0 / 0`. No numerical tests
or Ruff run for this Git synchronization; combined runtime behavior was
not tested. Backup patch and checksums are under
`/tmp/pepsy-sync-20261005-9NEKeT` and may expire.
