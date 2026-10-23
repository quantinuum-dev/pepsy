# 2026-10-23 — Shorten README and changelog

- Scope: simplify both entry points, commit and push synchronized `main` and
  `develop`, and finish on `develop`.
- Baseline: clean `develop` at `1717c1b`, matching local/remote `main` and
  `origin/develop`. This confirms publication of the previous version alignment.
- Commit status: this record accompanies the documentation commit; branch
  fast-forward and push follow the commit.

README now gives the package purpose, version, installation, and starting
guides in 30 lines rather than 49. The changelog gives version highlights in
65 lines rather than 1,768, retaining every version heading and date. Its
full-notes link targets the immutable pre-edit Git snapshot, preserving access
to detailed compatibility, numerical, and validation records. The migration
guide now calls the shortened changelog “release highlights.”

Validation: relative Markdown links and anchors, the README image, version
agreement, retained release headings, the full-note snapshot, and
`git diff --check` passed. Package version remains 0.6.0. No implementation
changes or numerical test run.
