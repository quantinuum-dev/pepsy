# 2026-09-26 — PepsSampler backend inference and to_backend

- Scope: implement the user's requested automatic Torch/JAX inference or
  explicit to_backend converter, and fix the studied sampler backend issues.
- Branch / baseline: develop / 80f451a.
- Commit status: working-tree edits only; nothing staged, committed, published,
  or installed. Existing qMERA/tree/roughening changes were preserved.

## Implemented

- PepsSampler now infers dense source backend/dtype/device, or converts a
  private copy with the user-supplied callable. Refresh follows the same rule.
- Native identity caps, density matrices, probability validation and draws;
  dtype-aware roundoff tolerance; lazy diagnostic scalar readout.
- Scoped JAX device context fixes identities and random draws on a source
  device other than the process default.
- Source preservation including mutating converters; Torch gradient
  preservation; one amplitude contraction per distinct final batch group.
- API documentation, changelog, and focused regressions updated.

Details, dependency classification, reproduction evidence, and limits:
[backend fix note](../docs/development/notes/peps_sampler_backend_fix.md).
Earlier study: [Torch study](2026-09-26-peps-sampler-torch-study.md).

## Validation

- Final PEPS sampler suite with two JAX CPU devices: 35 passed.
- Torch CPU/CUDA complex128/complex64 reproduction matrix: 12/12 passed,
  including exact, Quimb, DMRG/FIT, and identity futures.
- Broader sampler/public API/package layout selection: 195 passed, 1 failed.
- Final smoke: 152 passed, 1 failed.
- Both failures are the unchanged package-version check: installed
  metadata/runtime 0.4.0 versus pyproject 0.5.0.
- Ruff src/tests and whitespace checks passed.

## Limits

Results and prefix grouping remain Python/NumPy; validation and compression
retain scalar synchronizations. JAX GPU and distributed/sharded arrays and
minimum dependency versions were not tested. No production speedup is claimed.
No requested implementation remains blocked.
