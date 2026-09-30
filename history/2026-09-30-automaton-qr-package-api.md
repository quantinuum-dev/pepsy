# 2026-09-30 — Integrate automaton plus QR into Pepsy

- Scope: corrected user request for package-level integration, beyond the
  preceding notebook-only composition.
- Branch / baseline: Pepsy `develop` at `c1ff0f8`.
- Status: uncommitted working-tree edits; nothing staged or published.

Added `preparation` and explicit `delinearize`/`delinearize_opts` to both
cluster-MPO facades; added optional numerical reduction to channel-plan
evaluation and tracing. Reused existing constructors and QR implementation.
Reports separate exact channel dimensions from the final numerical result.
Corrected exact structural sharing being reported as reference QR projection.
Updated owning API guides, module map, changelog and status ledger.

The final affected domain/API/layout selection passed **297 tests** after
the report correction. Ruff, local links and whitespace checks passed.
The six-site p=5 integrated OBC/PBC probes reproduce the previous notebook
bond dimensions. No full repository suite, no new notebook edits.
See [implementation, checks and limits](../docs/development/notes/2026-09-30-automaton-delinearisation-api.md).

Preserved existing edits, including the concurrent trace/channel API audit.
The earlier notebook handoff remains a record of that narrower completed work.
