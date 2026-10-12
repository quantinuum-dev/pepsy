# 2026-10-06 — Publish PEPS optimizer updates

- Scope: user requested committing and pushing the PEPS work in Pepsy.
- Starting branch / baseline: `develop` / `8f7c896`.
- Publication scope: accumulated PEPS batching, unitary-target norm handling,
  sweep safeguards, global recovery/backend/normalization fixes, and JAX
  defaults plus SVD registration; associated tests, API docs, and evidence.
- Separate gradient-solver edits and unrelated historical handoffs are
  excluded and preserved in the working tree, including their changelog text.

The earlier handoffs describe their original uncommitted state. This entry
records the subsequent publication request. Validation evidence remains in
the [global fixes](2026-10-06-peps-global-mode-fixes.md),
[JAX defaults](2026-10-06-peps-global-jax-defaults.md), and
[JAX registration](2026-10-06-peps-global-jax-registration.md) records.
Those focused checks passed; they do not establish full-suite/GPU coverage.

Fetched `origin/develop` before publication: remote advanced to `7e49694`,
containing stabilizer work. Integrate that history without rewriting remote
commits, preserving both projects' changelog additions. Commit/push results
are reported by the session and Git history.
