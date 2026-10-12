# 2026-10-06 — Default to disentangling MPS stabilizer measurements

- Scope: user requested `disentangle=True` by default.
- Branch / baseline: `develop`, `d41ea9f`, already ahead of origin by two commits.
- Commit status: working-tree edits only; nothing staged, committed, or pushed.

## What changed

- `StabilizerMpsSimulator.measure`, `measure_many`, `measure_reset`, and
  `StabilizerMpsSampler` resolve omitted/`None` options to `True`. Certified
  native tableau collapse remains first; general basis-updating localization
  is now the fallback. Explicit `False` retains fixed-basis MPS collapse.
- Public keyword signatures retain `None` sentinels so the `absorb_basis`
  alias, including explicit `False`, still works without spurious conflicts.
- Stream dispatch, frame-layout tracing, and coalesced replay use the same
  default. Shared replay preserves other optimizers' explicit-`None` policy.
- Updated API docs, changelog, method documentation, and maintained skill.
  Fixed-basis regressions now select that representation explicitly. New tests
  compare NumPy/Torch collapses against dense projections and check native
  replay, general fallback, and tree replay compatibility.

## Validation

- Measurement/sampler/STN suites: **505 passed, 1 skipped**.
- Public API/package layout/trajectory/noisy dense-reference suites:
  **105 passed, 2 skipped**.
- Final shared-replay check (tableau measurement, trajectory execution,
  trajectory noise, tree stabilizer): **387 passed, 2 skipped**.
- Final replay selection repeated after reusing the existing STN recognition
  helper: **387 passed, 2 skipped** (8.33 seconds).
- Ruff, skill quick validation, catalog validation, and `git diff --check`
  passed; Ruff and diff checks also repeated after the final code edit.
- Initial full run aborted in Matplotlib's macOS GUI backend at
  `test_build_itf_lattice_show_returns_schematic_drawing` (exit 134).
  Final full suite with `MPLBACKEND=Agg`: **7214 passed, 526 skipped,
  717 warnings**, 1004.45 seconds. This run includes the final code and
  regressions. Temporary log: `/tmp/pepsy-disentangle-default-full-agg.log`.
- Reused the session's upstream audit in the unchanged `genpy` environment;
  dependencies and numerical kernels were not changed.

## Decisions / limitations

- Preserved the reviewed native shortcut, with no compilation cache or
  optimization rewrite. See [prior benchmark](2026-10-06-helix-no-shortcut-benchmark.md).
- No notebook or sibling-repository edits. Existing untracked handoffs remain
  preserved. General non-Clifford collapse may incur localizer work and can
  truncate at finite `chi`; explicit `False` remains available.
