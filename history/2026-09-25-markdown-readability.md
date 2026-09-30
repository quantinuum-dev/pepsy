# 2026-09-25 — Markdown organization and readability review

- Scope: review documentation organization and make Markdown concise and clear.
- Branch / baseline: `develop` / `7bde713`.
- Commit status: included in the documentation and MPS reporting cleanup commit.

## Changes

- Inventoried all 220 tracked Markdown files and reviewed the entry pages,
  indexes, installation/contribution instructions, and navigation in long APIs.
  This was an editorial and structural review, not a fresh numerical audit of
  every API statement.
- Revised 19 pages. Reduced README from 682 to 273 whitespace-delimited words
  and API starting points from 741 to 268. Moved backend compatibility detail
  to the package map instead of dropping it.
- Added clear contributor sections, task tables, and links from long API
  guides. Centralized installation guidance and used owning-namespace imports
  in the quickstart.
- Corrected two registry-style install commands to install extras from the
  checkout. Replaced four absent qMERA script references with existing examples
  and the qMERA API guide.
- Distinguished proposals and historical notes from supported behavior.
  Added a short Markdown writing guide to the development index.
- Preserved detailed numerical contracts, completed history entries, source
  implementation, workflows, and the release tag.

## Distribution decision

After the publication attempt, the user explicitly cancelled PyPI/TestPyPI
publication. The earlier release-publication handoff's resume steps are now
superseded. Pepsy remains distributed through GitHub; no registry retry or
account setup is authorized by those historical instructions. The installation
and development guides now reflect this decision.

## Validation

- All 378 checked relative Markdown link targets across the original 220
  files exist. Repository-source links in changed pages also resolve locally.
- All 30 checked section links in changed pages have rendered targets.
  Rendered entry pages contain their expected headings and navigation.
- Strict Sphinx HTML build passed with an empty diagnostic log after fixing
  one new tree-section slug mismatch. No warning suppression was added.
- README, quickstart, and getting-started Python examples all ran successfully.
- `git diff --check` passed. Full numerical suites and Ruff were not rerun for
  Markdown-only changes.
