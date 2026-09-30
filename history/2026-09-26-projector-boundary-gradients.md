# 2026-09-26 — Rank-deficient boundary-MPS derivatives

- User scope: correct the singular QR / biased adaptive backward in Gaugy's
  complete cluster-PEPO local cost, including actual boundary truncation.
- Branch/baseline: `develop`, `1d327e6`; companion Gaugy baseline `4cad51a`.
- Commit status: this entry accompanies the local fix; no push.

## Implemented

Added the package-owned `pepsy:projector` split and first-order composed
Torch VJP. `contract_flat` exposes `mps_factorization="projector"` for dense
2D direct boundary compression, using it at every canonicalization/reduction/
truncation stage. Existing default `"qr"`, global drivers, SU, and CTMRG are
unchanged. Cutoff/max-bond semantics use installed Quimb, then numerical-null
removal; undefined gaps/noninvariant cotangents fail explicitly.

See the [derivation, upstream audit, and limitations](../docs/development/notes/2026-09-26-projector-boundary-gradients.md)
and [public API](../docs/api/boundary/metrics.md). Native symmetry, JAX,
rank-changing differentiability, and higher derivatives are not claimed.

## Validation

- Focused projector/backend/API/layout/Torch-SVD selection: **135 passed**.
- Full Ruff `src tests`: passed. `git diff --check`: passed.
- Downstream new whole-loss regressions and actual optimization: 3×3 order3
  OBC/PBC, chi1/16; 4×4 t1/depth10 chi1/4/16. Complete production 4×4
  directional error <4.5e-10 initially and <1.8e-10 after optimization.
- Full Gaugy suite **534 passed**; unchanged reference notebook passed in
  215.6 s. The shared numerical-contract guide now records the paired-factor
  algorithm's scope separately from global Torch registration.
- Full Pepsy suite (`pytest -q -o addopts='' --maxfail=1`): **4,789 passed,
  121 skipped**, 731 warnings, 353 s; no early stop. Skipped unavailable or
  conditional optional coverage is not claimed as validated.

No installed dependency edits or generated benchmark files are committed.
