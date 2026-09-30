# 2026-09-24 — Lighter MPS imports and explicit ownership

- Scope: deeper package-structure review and a verified import-boundary cleanup.
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: uncommitted; nothing staged, published, or installed.

## Finding and implementation

The MPS package initializer eagerly imported layout, Gibbs, and replay code.
Consequently even an empty reserved diagnostics module loaded the simulator,
Gibbs preparation, and MPO machinery. Replaced those eager exports with the
existing package's lazy lookup pattern, retaining type-checking declarations,
all advertised exports, direct submodule imports, and the historically
accessible `gibbs` module attribute. Implementations and numerical policies
are unchanged.

Clarified that MPS `compression.py`, `normalization.py`, and `diagnostics.py`
are empty reserved import paths. Their implementations remain in
`optimizer.py`; their existence does not authorize an extraction. Updated the
[optimizer ownership map](../docs/development/modules/optimizers.md) and
[MPS API guide](../docs/api/optimizers/mps.md).

## Local measurements

Three fresh Python processes per import before and after the edit, same local
environment and checkout; medians below. Filesystem caches were warm and
timings are indicative, not portable performance guarantees.

| Import | Before seconds | After seconds | Loaded Pepsy modules before → after |
| --- | --- | --- | --- |
| `pepsy.optimizers.mps` | 0.9553 | 0.0214 | 41 → 4 |
| `pepsy.optimizers.mps.layout` | 0.9749 | 0.9527 | 41 → 23 |
| `pepsy.optimizers.mps.diagnostics` | 0.9920 | 0.0209 | 42 → 5 |

Layout still loads its numerical dependencies; its timing difference is too
small to claim a meaningful speedup. The robust improvement there is avoiding
unrelated replay/Gibbs/MPO imports. These results do not measure installation
size, simulation speed, or memory use.

## Validation

- Import boundaries, public API, package layout, MPS layout, Gibbs MPS, and
  sampler suites: **198 passed, 6 skipped**, with 13 warnings.
- New fresh-process tests verify light namespace/placeholder imports and
  layout independence; the layout schedule also survives pickle round-trip.
- Public export identity, star imports, `gibbs` access, and unknown-name errors
  are covered. Existing public names are preserved.
- Ruff, local Markdown link-target checks, and `git diff --check` passed.
- No numerical kernels or cross-subsystem implementations changed; the full
  numerical suite was not rerun. Prior numerical failures remain outside scope.

## Further findings, not implementation tasks

Other optimizer initializers (MPO, PEPS, tree, tree-PEPS, energy, qMERA) also
eagerly import implementations and need their own import-order and behavior
checks before applying this approach. Internal `tensors.core` compatibility
imports and large-module extraction remain separate work. Do not infer that
an empty proposed module already owns a working algorithm.
