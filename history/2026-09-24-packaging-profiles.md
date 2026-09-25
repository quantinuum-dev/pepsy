# 2026-09-24 — Packaging profiles and installation clarity

- Scope: continue the requested review and cleanup for a lighter, clearer package.
- Branch / baseline commit: `develop` / `64f7e1f`.
- Commit status: working-tree edits only; nothing staged, committed, or published.
  Earlier session changes remain intact.

## What changed

- [Dependency profiles](../pyproject.toml) compose existing extras: `vmc-torch`
  reuses `torch`, `vmc-netket` includes `symmetry`, `vmc` combines both VMC
  profiles, and `test-extended` combines its existing feature dependencies.
  Base requirements, extra names, and expanded constraints are unchanged from
  this session's starting state, including earlier upstream version updates.
- [Installation guidance](../docs/installation.md), the
  [README](../README.md), and [getting started](../docs/getting_started.md)
  distinguish base, development, and optional installs. Removed the stale
  machine-specific environment path from the installation page.
- [Package CI](../.github/workflows/ci.yml) now exercises a two-site MPS gate,
  dense-state comparison, and norm contraction in the base wheel environment.
  [Layout tests](../tests/test_package_layout.py) check composed dependency
  profiles, cycles, and separation of Torch from NetKet/JAX requirements.

## Validation in this session

- Package layout, public API, and import boundary tests: **69 passed**.
- `python -m ruff check src tests`: passed.
- Built baseline and updated wheels from temporary snapshots using the
  existing environment's Setuptools backend; no dependencies were installed.
- Compared built metadata: every extra's expanded external requirements and
  all five direct base requirements are preserved. All package payload bytes
  are identical between the two wheels.
- Offline pip dry-run for the updated wheel with `vmc,test-extended` resolved
  successfully using installed dependencies; its only proposed installation
  was `pepsy-0.4.1`. No installation was performed.
- Parsed CI YAML and ran its exact numerical smoke code against the extracted
  updated wheel with existing environment dependencies: passed. A fresh base
  installation remains a hosted CI check, not a local result claimed here.
- Relevant local Markdown links and `git diff --check`: passed.

## Findings and limits

- The updated wheel is 2,059,063 bytes, with 187 package files plus metadata.
  It contains no tests, documentation, history, or repository tooling.
  There is no obvious bundled-data bloat to remove. This cleanup reduces
  repeated configuration and clarifies optional installs; it does not produce
  a meaningful reduction in wheel size or numerical dependency footprint.
- No package implementation changed in this pass. The full numerical suite
  was not rerun. The earlier **4592 passed, 121 skipped, 13 failed** result and
  failure identities remain recorded in the
  [optimizer import handoff](2026-09-24-optimizer-import-boundaries.md).
- Further reductions in required dependencies or public compatibility aliases
  would need a separate compatibility decision and targeted validation.
