# 2026-10-08 — Fix failures from full repository recheck

- Scope: user requested fixes for the reported full-suite failures.
- Branch / baseline: `develop`, `ad8ed05`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.
- Prior PEPS and concurrent MPS/JAX edits preserved.

Implemented BP residual confirmation, early compact-channel backend binding,
unfused native CTMRG projector contractions, and explicit MPS boolean readback.
Corrected test precision/rank assumptions without relaxing tolerances or
changing library defaults. Also isolated the global JAX mesh installed by
NetKet imports, which was breaking later CPU-backed tests.

## Validation

- Uninterrupted full suite: **8,263 passed, 9 skipped**, zero failures,
  2,116 warnings, 2,726.62 s. All 8,270 collected IDs are accounted for;
  two of the nine skips are MPI modules skipped during collection.
- Focused BP suites: 134 passed; broader native/compatibility/BP selection:
  282 passed. MPS ledger parity: ten passed. NetKet followed by qMERA:
  16 passed, confirming the mesh isolation across suites.
- Ruff, whitespace, and skill catalog (12 skills) pass.
- Separate PEPS refinement edits appeared during the final run and were
  preserved. Their five closest suites passed **132 tests**, 33 warnings,
  no skips, in a fresh process; source/test hashes stayed unchanged during
  that run. This is separate validation of the later working-tree changes.

No unresolved failure from the reported recheck remains under the documented
test configuration. Skipped MPI/multiple-device/Metal configurations remain
unvalidated. No implementation changes were made after starting the final
full run by this task; the separate PEPS edits are identified above.

See the [detailed correction and validation record](../docs/development/notes/2026-10-08-full-suite-failure-corrections.md)
for upstream evidence, numerical probes, test results, and remaining work.
