# 2026-10-13 — MPS mode discovery and long-support DMRG alignment

- Branch: `develop`, working tree only; no commits or publication.
- Added public `MpsOptimizer.supported_modes()` so downstream adapters consume
  native replay names without copying a whitelist.
- Fixed DMRG sub-MPO support/guess windows for more than two support sites;
  no target truncation, normalization or schedule policy was changed.
- Gaugy now accepts native target/V modes, preserves named DMRG schedules and
  permutation logical order, and exposes independent target numerical options.

Fresh validation: 226 native MPS tests, 23 selected Symmray/fermionic/sub-MPO
checks and 55 API/layout tests passed. Gaugy's 410 scoped replay tests and four
actual-cell tests passed; its reduced DMRG notebook executed successfully.
Pepsy Ruff and whitespace checks passed. An unrelated flat-PEPS failure was
reproduced with pre-task configuration; full suites were not run.
See the [audit and scoped evidence](../docs/development/notes/2026-10-13-mps-mode-alignment.md).
Concurrent unrelated working-tree edits and saved notebook outputs are preserved.

## Requested publication scope

The user authorized committing and pushing the reviewed MPS work on
2026-10-13. The selected Pepsy commit contains native mode discovery,
arbitrary-support DMRG windows, their tests, owning guide, status row and
only the MPS changelog entries. Concurrent PEPS/VMC/boundary work is excluded.
Fresh pre-commit validation: **180 compression-mode/API/layout checks passed**
in 8.25 seconds; Ruff over `src tests` and whitespace checks passed.
Earlier uncommitted statements above describe their recorded baseline.

The isolated staged tree also passed the same **180 checks** in 7.89 seconds.
That copy contains only the selected commit content over the current committed
baseline, excluding unrelated working-tree edits.
