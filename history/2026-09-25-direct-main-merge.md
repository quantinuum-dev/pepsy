# 2026-09-25 — Direct main/develop integration

- Scope: user requested a direct branch merge, keeping `main` and `develop`
  and ending the draft-PR workflow for this integration.
- Parents: `main` at `e652197` and `develop` at `eccdfcd`.
- Commit status: this handoff accompanies the direct merge commit. Both
  branches are to point to that commit, with `develop` checked out for work.

## Merge and validation

- Refreshed origin and confirmed these were the current remote tips. Both
  local and remote repositories had only `main` and `develop` branches.
- The local merge was conflict-free and retained main's boundary alias
  cleanup. Before adding this handoff, its complete tree was
  `c095111246ce35959c4ebc0450b976773bb4b4f5`, identical to GitHub's tested
  merge `2a70bf7a4e866c45b41e99ccb1cb90c2231f6186`.
- [PR merge CI 36154338426](https://github.com/quantinuum-dev/pepsy/actions/runs/36154338426)
  and [develop CI 36154307048](https://github.com/quantinuum-dev/pepsy/actions/runs/36154307048)
  both completed successfully before the direct merge. The only addition to
  that tested merge tree is this Markdown handoff; no new numerical tests
  are required for it.
- Local API/migration/boundary validation and documentation limitations are
  recorded in the [release-preparation handoff](2026-09-25-release-readiness.md).

## Publication and remaining work

- Push the direct merge to `main` and fast-forward `develop` to the same
  commit. Close [draft PR #12](https://github.com/quantinuum-dev/pepsy/pull/12)
  if GitHub does not automatically close it when the merge is pushed.
- Keep both branches; no branch deletion, version bump, tag, or package
  publication is part of this task. Release version selection remains open.
