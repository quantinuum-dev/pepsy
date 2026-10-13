# 2026-10-13 — Large starting caps with bounded failure recovery

- Scope: finish and validate the pending PEPS ALS adaptive-cap work, keeping
  healthy fits free of repeated chi searches.
- Branch / baseline: `develop` / `43dfc4a`.
- Status: working-tree changes only; nothing staged, committed, or pushed.
  Preserved the existing norm-diagnosis record and pending implementation.

Strip ALS now defaults to `boundary_convergence="on_failure"` through `auto`.
Start at the componentwise maximum of configured fitting, normalization,
and evaluation caps. Keep each attempt fixed; existing invalid norm,
overlap, fidelity, or initial ALS diagnostics request an exact doubling of
the affected component. Ambiguous fidelity violations double both. Restore
the failed batch, discard old boundary guesses, and retry. Retain larger
caps across batches and calls. A doubling above the ceiling raises
`BoundaryChiLimitError`, preserving earlier accepted batches. Default limits
are eight times the resolved starting caps; explicit limits take precedence.

This session corrected explicit metric `chi=None` preservation, validated
unknown target norms with monitoring disabled, moved final onsite
normalization inside batch recovery, and inspected requested raw final
diagnostics even when a finite best iterate exists. Fixed metric FIT caps
and supplied boundary handles now reject ineffective adaptive recovery before
state changes. Public documentation describes controls and cost.

No routine convergence probes run. Compared with the known-unit-norm shortcut,
one target self-overlap is measured per batch attempt and reused for fitting
and the postcheck. Poor but admissible fidelity and invalid solver proposals
alone do not request growth. Existing validity tolerances remain unchanged.
This policy detects failures; it does not certify in-range metric accuracy.

## Validation

Used the existing isolated validation environment with one BLAS/OpenMP thread.
Installed versions: Quimb `1.15.1.dev91+gb3bf45a50`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Cotengrust `0.2.1`,
Symmray `0.4.1.dev18+g44fc5ae53`, Torch `2.14.1+cu126`, JAX `0.11.2`.
Inspected installed PEPS metric/normalization and Quimb boundary signatures,
compression option validation, and boundary ownership.

- Initial recovery tests: **22 passed, 1 failed**, exposing overwritten exact
  readout `chi=None`; corrected without weakening the assertion.
- Broad sweep/strip/driver/batching/convergence/full-update/public API selection:
  **776 passed**, before the last optional-diagnostic and handle-validation
  refinements. Log: `/tmp/pepsy-growth-regression.log`.
- Subsequent growth selection: **36 passed**, including NumPy/Torch recovery
  from genuinely truncated 3×3 boundaries and fresh fixed-cap equivalence.
  Starting caps `(1,2)` grow to `(4,8)` on this deterministic fixture.
  Log: `/tmp/pepsy-growth-final-focused.log`.
- Final growth/defaults/strip-review/public API/package-layout selection:
  **143 passed**, after the final refinements. Log:
  `/tmp/pepsy-growth-final-validation.log`.
- Ruff on `src tests`, changed-document link checks, and `git diff --check`
  passed. These selections do not constitute a full-suite pass.

Consulted official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray array documentation URL was unavailable. Classification:
**defer** upstream changes; recovery uses current installed contraction and
FIT APIs without changing dependencies or dispatch behavior.

No stopped benchmark was restarted. A full 4×4 higher-D trajectory with this
new recovery policy remains unvalidated. Existing controlled trajectory and
norm-error evidence remains in the [earlier handoff](2026-10-13-peps-als-norm-diagnosis-rerun.md).
