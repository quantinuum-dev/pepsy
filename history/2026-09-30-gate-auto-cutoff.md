# 2026-09-30 — Move automatic gate cutoffs into Pepsy

- Scope: user requested that automatic dtype-based cutoffs belong in Pepsy's
  `gate_simple` and general `gate`, rather than the roughening example.
- Branch / observed baseline: develop / f01f596. Other task changes were
  already present, including SU exponent tracking and MPS mode work.
- Commit status: this task's edits are uncommitted; nothing staged or pushed.

Public gate APIs now default to auto cutoff/mode and resolve the shared MPS
policy before upstream dispatch. Explicit numeric cutoffs remain supported.
The roughening runner passes the options through and has no private policy
import or local resolver. Updated API docs, changelog, runner docs, and tests.

Fresh checks: 179 gate/SU tests and 40 downstream PEPS tests passed. Public
API/layout had 53 passes and the known installed-version mismatch failure.
Package Ruff and diff checks passed. No full-suite or GPU claim.

See the [implementation and dependency audit](../docs/development/notes/2026-09-30-gate-auto-cutoff.md).
