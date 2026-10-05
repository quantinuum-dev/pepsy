# 2026-10-05 — MPS trajectory publication

- Scope: user requested committing and pushing the completed MPS work.
- Branch / starting commit: `develop`, `2039fbc`.
- Publication: prepared for the authorized commit and push to `origin/develop`;
  this entry accompanies the implementation commit, identified by Git history.

The commit includes the [precision and memory-budget work](2026-10-05-mps-precision-memory.md)
and [prefix continuation, GPU gate/SVD batching and MPI correction](2026-10-05-mps-frontier-batching.md),
their tests, API documentation, scoped changelog entries and validation notes.
The earlier handoffs' uncommitted status describes their validation time.
Unrelated MPO, gradient, tree, sampling and cluster-status edits remain outside
this commit, including their changelog entries and existing history files.

Validation is reused from the unchanged implementation: 933 passing domain
tests, 32 passing slow cases, 44 passing gate-batch tests, 89 passing final
branch-bound checks, and 32 passing MPI cases per rank with both two and three
ranks sharing one GPU. These selections overlap and are not summed. Ruff,
whitespace and documentation-link checks passed. The full-suite attempt still
stops at the three previously reproduced baseline BP failures. See the linked
handoffs for skips, commands, detailed evidence and implementation limits.

## Upstream integration and publication checks

The scoped implementation commit is `ad0d628`. Fetch found the new upstream
commit `8f912a9` (reusable Stim streams and backend-aware stabilizer replay),
which merged without conflicts. Only the unrelated changelog hunks were
temporarily saved for the merge; all unrelated source edits stayed untouched.

An exported staged-tree snapshot excluded every unrelated working-tree edit.
Fresh checks on that combined snapshot passed **367 tests, 2 slow deselected**:
MPS trajectory/continuation/memory/gate batching and precision, MPI backend
fingerprinting, shared trajectory replay, upstream Stim stream coverage, and
public API/package layout. All-source/test Ruff, staged whitespace, clean
snapshot import-path checks and local documentation links also passed.
The tests took 39.42 seconds; log: `/tmp/pepsy-mps-publication-check.log`.
This is a focused post-merge check, not a fresh full-suite run.
