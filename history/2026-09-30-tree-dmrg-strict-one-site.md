# 2026-09-30 — Require one-site Tree DMRG

- Scope: user accepted strict one-site Tree `dmrg`, consistent with MPS,
  with `dmrg1` retaining its deprecated-alias warning.
- Baselines: Pepsy `develop` / `775d405`; examples `main` / `efc4a3d`.
- Status: working-tree edits, not committed or pushed. Unrelated sampler,
  PEPS reference, documentation and notebook edits preserved.

Construction, replay (including shots), and effective block resolution now
reject larger blocks for `dmrg`/`fit`/`dmrg1`. Invalid replay leaves the
parent's configuration, gate queue and state unchanged. `dmrg2`/`dmrg3`
retain their schedules. Roughening performs the same early configuration
check. API/changelog and runner documentation explain the restriction.
See [implementation and compatibility evidence](../docs/development/notes/2026-09-30-tree-dmrg-strict-one-site.md).

Fresh checks in the shared Python 3.12 environment with local source,
CUDA hidden and OMP/OpenBLAS threads one:

- Focused optimizer/API selection: **382 passed**.
- Full tree-domain plus public API/layout: **1543 passed, 101 skipped,
  3 failed**. Two composition tests still expected switching into `dmrg`
  with block size three; updated these to require rejection and then use
  `dmrg3`, preserving the original settings-retention check. Final complete
  composition module rerun: **25 passed**. The remaining failure is the
  known installed-distribution 0.4.0 versus source 0.5.0 mismatch in
  `test_package_version_matches_installed_distribution`. The full domain
  selection was not repeated after these test-only corrections.
- Examples run/roughening/sweep/entrypoint/layout selection: **330 passed,
  2 skipped, 4 known bubble failures**: defaults quality tracking plus
  the complex64 long-range `fit`, `dmrg`, and removed-MPS-`dmrg1` cases.
- Package Ruff, changed examples Ruff, relevant existing documentation
  links and whitespace checks pass. Broad examples Ruff retains six
  existing effective-model E402/E731 findings.

Logs: `/tmp/tree-dmrg-strict-focused.log`,
`/tmp/tree-dmrg-strict-suite.log`, `/tmp/tree-dmrg-strict-compositions.log`,
and `/tmp/roughening-tree-dmrg-strict.log`. No GPU speedup measurement,
dependency changes, notebook execution or production runs.
