# 2026-11-19 — Publish canonical DMT metadata and mapped-2D benchmarks

- Authorization: pull Pepsy, commit the completed work, and push.
- Branch / starting revision: `develop`, `04a712435391712b47ad42777bdb4829d9dd7519`.
- Pull: `git pull --ff-only` reports already up to date; no merge or conflict.
- Publication: this entry accompanies the DMT publication commit on develop;
  its revision is available in Git history. Push target is `origin/develop`.
- Device-local `AGENTS.override.md` is excluded from staging/publication.

The commit includes the completed 4×4/6×6 mapped-MPO research records,
canonical interval prototype, default persistent ordinary-MPO DMT replay,
independent Pauli metadata ownership, regression tests, examples, API and
skill-reference updates. Earlier entries marked uncommitted describe the
state before this publication request.

Evidence is retained in the notes for
[2D scaling](../docs/development/notes/2026-11-19-dmt-2d-scaling.md),
[interval experiments](../docs/development/notes/2026-11-19-dmt-2d-window.md),
[persistent caching](../docs/development/notes/2026-11-19-dmt-canonical-cache.md),
and [metadata ownership](../docs/development/notes/2026-11-19-dmt-canonical-info.md).
Curated plot/data fixtures follow the existing development-note convention;
temporary checkpoints, profiles and large core dumps remain outside the repo.

Latest numerical validation includes 454 broad Pauli/MPO passes (one skip),
31 focused metadata passes and six ordinary canonicalization/rollback passes.
The public API/layout selection has 54 passes and the existing installed
version mismatch (0.5.0 versus checkout 0.6.0). Full-suite and GPU coverage
are not claimed. The documented chi192 numerical/native failures remain
unresolved and are not promoted to valid benchmark references.

Pre-publication Ruff, skill catalog and whitespace checks pass; the pull
introduced no source changes requiring another numerical run. No dependency,
installed-library or environment metadata changes were made.

Staged review normalized exported SVG whitespace and CSV line endings without
changing scientific values. The preceding publication journal is preserved
under its original filename.
