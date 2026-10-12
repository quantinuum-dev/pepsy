# 2026-10-08 — Full update and row/column refinement review

- Scope: user requested a careful review and report of full update followed by
  row/column sweep refinement. No implementation fixes requested.
- Branch / baseline: `develop`, `ad8ed05`, including the existing uncommitted
  PEPS review corrections. Other existing changes were preserved.
- This session adds only this handoff; no implementation/test edits,
  staging, commits, or publication.

## Confirmed finding — P2: refinement discards the overlap cap

In `src/pepsy/optimizers/peps/optimizer.py:2592`, block calibration returns
the norm/overlap pair. The driver takes only its first component and passes
that scalar to `_full_update_refine`. `_strip_update.py:80` uses it for all
three networks (candidate norm, overlap, target norm). With fixed boundaries,
the driver similarly uses only `boundary_chi[0]`.

Thus a larger requested/calibrated overlap cap does not control the
refinement objective or acceptance metric. A saved successful calibration
does not validate the lower-cap overlap actually used for this fit.
This does not imply every such update has an inaccurate objective.

Reproductions using Torch complex128, seed 24, D2, direct boundaries,
zero cutoff, four pair ALS iterations and two strip passes:

- Normalized 3x3 PEPS, RZZ diagonal `exp([-.3j,.3j,.3j,-.3j])` on
  `((1,0),(1,1))` and `((1,1),(1,2))`. Adaptive start `(4,32)`,
  maximum `(32,64)`, patience 1: the block calibration selects `(8,36)`;
  instrumented refinement calls use `8` for all three networks.
- Fixed `boundary_chi=(4,32)` uses `4` for all three networks.
- On a normalized 4x4 PEPS with the same gate on `((1,1),(1,2))`, changing
  only the overlap cap from 4 to 32 leaves the reported refinement
  infidelity unchanged at `0.042080461043959905`. Dense-reference
  infidelity is `0.041949705913109314` in both runs.

Suggested correction: carry separate norm/overlap caps into refinement
and retain the matching checked overlap handle when possible, or explicitly
calibrate the common scalar cap that refinement actually uses. Add a
regression covering unequal caps. This finding remains unfixed.

## Implementation assessment

- Two-site full update performs QR/LQ reduction, positive norm projection,
  optional environment gauges, shared native ALS, and rank-limited
  reconstruction with exterior tensors fixed.
- `refine_sweeps > 0` enables a fixed-rank one-site variational strip fit
  against the exact accumulated gate-block target. It is separate from
  DMRG boundary compression and from Hamiltonian ground-state DMRG.
- One pass visits each site once; successive passes reverse direction.
  Refinement defaults to disabled. The gate order and block-closing rules
  determine whether a block spans the full row/column gate layer.
- Existing checks cover exterior locality, tensor metadata, backend/device,
  rank preservation, mixed gate conversion, smart scheduling, cache
  invalidation, normalization/exponents, and invalid-candidate rollback.

## Fresh validation

Activated the existing `../envs/py312`, with bounded BLAS/OpenMP/Numba
threads. Confirmed local source imports and installed public Quimb ALS
signature. Versions: Quimb 1.15.1.dev90+g6a3906cbe, Autoray
0.11.1.dev14+g014a3f69a, Cotengra 0.8.3.dev8+g8954240f2,
Symmray 0.4.1.dev15+g0374aaa3c, Torch 2.6.0+cu124, CuPy 14.1.1.

- Full-update, strip-refinement, CuPy full-update, gate-order,
  smart-gate-order, and environment-reuse modules: **130 passed**, 33
  warnings, 93.66 s, no skips.
- Boundary-convergence and shared BP reduced-update modules: **72 passed**,
  15 warnings, 13.10 s, no skips.
- Combined: **202 passed**. An initial collection command used an incorrect
  environment-reuse filename and ran no tests; the corrected command above
  completed successfully.
- Additional 3x3 D2 complex non-diagonal two-gate row and column probes
  matched reported block infidelity to dense references to about 1e-15.
  Three passes lowered normalized cost from 0.256663 to 0.196592 (row)
  and 0.261140 to 0.218919 (column). With final normalization disabled,
  both satisfied `<candidate|candidate> = <candidate|target>` to roundoff.
- Additional strip refinement with `fit_mode='eff'` and `'dmrg2'`, two
  boundary iterations and two strip passes, was accepted and matched the
  dense infidelity `0.02894231189192` to about 1e-14.
- `git diff --check` passed. No fresh full-repository suite, lint run,
  long-time evolution benchmark, or large-D accuracy certification.

Temporary evidence: `/tmp/pepsy_full_strip_review_tests.log`,
`/tmp/pepsy_full_strip_review_shared.log`,
`/tmp/pepsy_full_strip_review_probe.py` and its `.log`, and
`/tmp/pepsy_full_strip_review_edges.py` and its `.log`.
