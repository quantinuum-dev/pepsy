# 2026-09-25 — Complete the status-review maintenance follow-up

- Scope: the user requested completion of the remaining work identified in
  the [status review](2026-09-25-status-review.md).
- Branch / baseline: `develop` / `9c16467`; remote `main` was `46eb601`.
- Commit status: cleanup is committed locally on `develop`. Remote branch
  promotion is blocked pending explicit user approval; no push or merge ran.

## Changes

- Corrected the roadmap's claim that 0.4.1 is current. Its workstreams are
  now clearly a dated snapshot, with links to the changelog and migration
  guide for released behavior.
- Added `pytest -ra` to routine, full, and MPI CI commands. This exposes
  skipped-test reasons without changing selection, coverage thresholds, or
  numerical behavior. The contributor guide explains how to assess skips.
- Clarified the existing GitHub Actions artifact distribution policy.
  A missing GitHub Release entry is intentional under this workflow, not a
  failed publication. Version 0.5.0 and its existing tag are preserved.
- Added the preceding review's live hosted validation evidence to history;
  it resolves the earlier pending full-workflow result.

## Local environment repair

- Installed missing mypy 2.3.1 and its dependencies; upgraded NetKet from
  3.21.0 to 3.22.4 after reviewing pip's dry-run plan. JAX remains 0.8.2;
  existing numerical dependencies were not upgraded.
- Installed missing documentation dependencies: sphinx-autoapi 3.8.1,
  Furo 2025.12.19, and sphinx-basic-ng 1.0.0b2. Existing Sphinx remains 9.1.0.
- Pepsy's core, VMC, Torch, symmetry, and documentation direct requirements
  satisfy their installed-version constraints in the designated environment.
- The unrelated Notebook/JupyterLab constraint mismatch reported previously
  remains outside this Pepsy repair. No sibling project or notebook package
  was changed.

## Validation

- VMC API and NetKet flat-Z2 tests: **57 passed**.
- Focused mypy: passed both source files. Ruff and the 12-skill catalog passed.
- Strict Sphinx HTML build: passed with an empty diagnostic log; **729**
  rendered local links, including fragments, resolved across the two changed
  documentation pages.
- MPI integration: **25 tests passed per rank** with both two and three ranks.
- All three workflow files parsed; embedded shell scripts passed `bash -n`.
- Full local suite after the environment repair: **4,607 passed, 121
  skipped**, 728 warnings, in 337.66 seconds. The 121 skips comprise 53
  CuPy, 42 CUDA, one Metal, and 25 single-process MPI cases. The MPI cases
  passed in the separate multi-process runs above.
- The focused Metal-ledger check outside the sandbox passed **all three
  cases**, including the actual Metal device. Its full-suite skip reflected
  sandbox GPU access. The 95 CUDA/CuPy cases remain unvalidated on this Mac.
- No numerical implementation or public API was changed. Whitespace and
  changed-page local file links passed.

## Boundaries

Automatic approval review rejected the attempted fast-forward and atomic push
of `main` and `develop`: it judged the request to finish remaining work
insufficiently explicit authorization to update the remote default branch.
The command was rejected before execution. Local `main` and both remote
branches retain their pre-task tips. The local cleanup is ready for approval
of the branch promotion; no workaround or partial remote push was attempted.

The large-module extractions remain separate proposed refactors; this pass
does not choose a numerical subsystem for restructuring. GPU and unavailable
upstream capabilities cannot be established by installing ordinary test tools.
No new release, tag movement, or registry publication is part of this cleanup.
