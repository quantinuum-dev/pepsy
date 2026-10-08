# Full-update cap correction and broader refinement assessment

Date: 2026-10-08. Branch: `develop`, baseline `ad8ed05`, with existing
uncommitted corrections preserved. The cap correction below is implemented;
larger-region refinement and solver changes are measured prototypes or
proposals, not new public modes.

## Implemented: retain the refinement contraction policy

`PepsOptimizer` now passes the full fixed/adaptive norm-overlap cap pair to
`refine_strip`. Candidate and target norms use the norm cap; the overlap uses
the overlap cap. The report retains scalar `chi` and adds `overlap_chi`.
Adaptive fitting receives the checked overlap handle and the corresponding
renamed target. Relabeling the candidate bra refreshes source signatures, so
cuts with incompatible index names are rebuilt rather than incorrectly reused.

The previous reproduction now uses `(4,32,4)` for candidate norm, overlap,
and target norm in fixed mode, and `(8,36,8)` following calibration at `(8,36)`.
Four Torch/CuPy regressions cover fixed/adaptive caps, handle handoff,
agreement with fresh uncached fitting, dense-reference fidelity, and output norm.
Scalar-cap callers retain their original behavior.

## Literature and upstream inspection

Reread [Lubasch et al., arXiv:1405.3259](https://arxiv.org/pdf/1405.3259),
particularly III B and Appendix B, and visually inspected Figs. 11, 12,
19, and 20. The stability prescription combines Hermitian/positive projection,
cutoff pseudoinverses, environment-derived gauges, internal QR/LQ regauging,
and normalization with tensor-scale balancing. Appendix B extends conditioning
to full tensors using separate site environments rather than a huge joint
pair matrix. Reduced updates hold the outer QR/LQ factors fixed, limiting
their variational space. These observations support examining full-tensor
refinement; they do not guarantee global optimality of any ALS sweep.

Checked the official [Quimb FullUpdate API](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tn2d/tebd/index.html#quimb.tensor.tn2d.tebd.FullUpdate)
and installed `quimb/tensor/tn2d/tebd.py` and `quimb/tensor/fitting.py`.
The installed FullUpdate fits complete plaquette site tensors, supports
periodic environment refresh, and offers tensor/bond conditioning. Its default
ALS uses direct solves with positive enforcement disabled. Our reduced pair
routine has a smaller search space; our strip routine already varies full
site tensors. Therefore “both are full update” does not imply identical
optimization problems. Quimb's refresh scheduling is not equivalent to our
source-validated cache reuse.

Installed versions and signatures were checked: Quimb
1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a, Cotengra
0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev15+g0374aaa3c,
Torch 2.6.0+cu124, CuPy 14.1.1. No dependency changes or shims.
The repository-required Quimb/Autoray/Cotengra/Symmray upstream pages were
checked. The Symmray array page and Quimb source page were unavailable to the
web tool; official repositories and installed source supplied the fallback.
PDF screenshots also failed in the web tool; downloaded PDF pages were
rendered locally for diagram inspection.

## Current implementation: stability opportunities

These are code-level observations and proposed experiments, not additional
behavior changes in this patch:

- **Prototype: supported-subspace solves.** Quimb's generic ALS used by our
  default pair and strip solvers floors eigenvalues at
  `lambda_max * pos_smudge`. It does not discard those directions. For
  `N=diag(1,0)` and `b=(1,epsilon)`, flooring at `r` gives
  `x=(1,epsilon/r)`, whereas a truncated pseudoinverse gives `(1,0)`.
  This illustrates why inconsistent overlap noise in a nearly null direction
  can produce large tensors. A future native PSD-pseudoinverse option should
  report retained rank and discarded RHS weight and retain current rollback
  checks. No realistic instability rate was measured here. Our QR fallback
  cuts singular values of a weighted design; its `rcond` has different
  spectral meaning from an eigenvalue floor.
- **Prototype: full-site preconditioning.** `_strip_update.py` Hermitianizes
  local matrices but has no environment gauge or inter-site QR/LQ conditioning.
  `_full_update.py` already has independent environment-root gauges, a gauged
  SVD initial guess, and final bond balancing. The shared weighted-QR route
  regauges after each local solve; the generic Quimb route does so after each
  complete pair sweep. Test strip preconditioners before adding more expensive
  multi-site local solves. A temporary variable-basis transformation must also
  transform the RHS; changing only the norm matrix changes the problem.
- **Prototype: physical-state-preserving scale control.** Network exponents
  protect global magnitude but do not bound every local tensor's entries.
  Track tensor-scale spread and local support conditioning. Any balancing
  should preserve the state and update/invalidate both norm and overlap
  caches. Do not silently rescale exterior arrays during a local fit.
- **Defer: unrestricted large joint solves.** Full-site matrices already grow
  as `D**4` by `D**4` for an interior site. Joint unstructured patches worsen
  this quickly. Matrix-free methods or two-site enrichment require separate
  conditioning, memory, and convergence studies.

## Small imaginary-time comparison

Normalized 3x3 D2 Torch complex128 PEPS, seed 24, one gate
`exp(-0.1 H)` on `((1,0),(1,1))`, where
`H = ZZ + 0.4 XX + 0.3 XI + 0.2 IZ`. Boundary cap 64, 20 pair/Quimb
iterations, tolerance 1e-10. Dense normalized fidelity is the common metric;
Quimb and Pepsy use different initializations and variational spaces.
Quimb conditioning and pre-normalization were disabled for this scoped
comparison; its end-of-gate rescaling makes raw-amplitude comparison unsuitable.

| Method | Exact target infidelity |
| --- | ---: |
| Pepsy reduced-pair FU | 0.006692180 |
| Quimb full-tensor FU, direct solve | 0.006167617 |
| Pepsy FU plus two strip passes | 0.006022586 |

Quimb with `als_enforce_pos=True` failed on this complex Torch input with a
complex/real matrix dtype mismatch in its positive reconstruction. No installed
library was patched. A separate attempted `imag=False` construction was rejected
by the installed class as untested. These are environment-specific limitations,
not a claim about all Quimb versions or all backends. Generalized upstream FU
replacement is deferred; the existing public generic ALS integration remains.

## Measured prototype: one fixed target for a bounded layer

The current local strip target starts from the already truncated state at
the beginning of that strip block. Later strips cannot use it to recover an
earlier block's discarded error. The prototype instead freezes one target
before a complete bounded layer, uses FU as a warm start, then calls the
existing native strip fitter across all rows and all columns against that
same target. It needs no joint row tensor or new dense multi-site solver.

Three normalized random 3x3 D2 Torch complex128 states, seeds 7/24/45.
The layer contains RZZ gates on all horizontal edges, RX at `(1,1)`, then
RZZ on all vertical edges. Gates are `diag(exp([-.3j,.3j,.3j,-.3j]))` and
`exp(-.17j X)`. Each edge appears once; the exact target has bond dimension 4.
Pair ALS uses 20 iterations/1e-10; boundary cap 64 and direct zero-cutoff
compression. One fixed-target cycle visits three rows and three columns,
one local pass each; a second cycle reverses the strip order.

| Seed | FU | FU + existing two-pass local refinement | FU + one fixed-target cycle | FU + two cycles |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.203550 | 0.190848 | 0.137554 | 0.130346 |
| 24 | 0.236235 | 0.227674 | 0.161627 | 0.154596 |
| 45 | 0.177893 | 0.170070 | 0.114349 | 0.108482 |

Entries are independently contracted global infidelities against the fixed
layer target, not products of gate fidelities. One-cycle results reduce
infidelity by about 28–33% relative to existing local refinement. Single-run
CPU timings for existing refinement were 0.557/0.585/0.557 s; FU plus one
fixed-target cycle including target construction took 0.629/0.668/0.602 s.
These small, single-thread measurements are not GPU/large-D scaling evidence
or a general speed claim. Both norm and overlap boundary caches recorded
4 hits/8 rebuilds after one cycle and 10 hits/14 rebuilds after two.

## Proposed next implementation

1. Reuse existing exact automatic-batch construction to freeze a bounded
   layer/window target. Never truncate this target to fit a batching budget;
   repeated bonds can force an earlier window boundary.
2. Use sequential FU as the initial state, then run native full-tensor ALS
   over rows and columns against that single target. Existing ordinary sweep
   mode already has global batch targets; the missing composition is the
   FU warm start with this native strip fitter.
3. Maintain distinct candidate-norm and target-overlap boundary stores at
   their calibrated caps. Across a directional pass, update the completed
   side incrementally and reuse the untouched side. Cache target-only norm
   work while its target and policy remain unchanged.
4. Within each strip, maintain moving prefix/suffix environments. Current
   caches avoid repeated contractions but still revisit prefixes/suffixes
   for signature checks. A sweep cursor should validate/extend only the
   necessary frontier, avoiding quadratic cache-lookup walks. Keep mutation
   invalidation for external writes and policy changes.
5. Use full-layer checkpoints, a consistent objective, and an independent
   higher-cap acceptance check when requested. Stop on negligible layer
   improvement. This is a bounded-work variational refinement, not a proof
   of a global PEPS optimum.
6. Only if this stalls, benchmark residual-selected overlapping 2x2 patches
   or controlled two-site enrichment against equal-time extra row/column
   cycles. Require data before choosing those more expensive paths.

Classification: **adopt** the cap correction; **prototype** shared-target
native refinement, support-aware solves, and full-site conditioning;
**defer** a Quimb replacement, stale-environment scheduling, and unrestricted
joint patch optimization. No compatibility shim was needed.

## Validation and artifacts

- Changed-path selection: **102 passed**, one warning, 23.97 s.
- CuPy FU, gate ordering, environment reuse, adaptive boundaries and shared
  reduced ALS: **104 passed**, 47 warnings, 20.03 s.
- Total **206 passed**, no skips. Ruff (`src tests`) and `git diff --check`
  passed. No fresh full-repository suite or long-time benchmark.
- API and changelog describe the implemented cap behavior. Existing unrelated
  edits were preserved; nothing staged, committed, or published.

Temporary scripts/logs: `/tmp/pepsy_strip_cap_fix_tests.log`,
`/tmp/pepsy_strip_cap_fix_shared.log`, `/tmp/pepsy_strip_cap_fix_probe.log`,
`/tmp/pepsy_quimb_fu_comparison.py` and `.log`, and
`/tmp/pepsy_layer_refinement_benchmark.py` and `.log`.
