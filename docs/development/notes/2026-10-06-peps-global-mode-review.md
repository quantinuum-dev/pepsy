# 2026-10-06 — Review PEPS global optimization

Scope: read, test, and report on `PepsOptimizer(mode="global")`; no fixes
authorized in this turn. Branch `develop`, baseline `8f7c896`; existing
working-tree changes preserved. Used the maintainer router, current global
API guide, implementation, focused tests, and installed Quimb TNOptimizer
source. Reused the unchanged environment/upstream audit from this session.

## Current operation and defaults

The driver builds the exact gated target, constructs a compressed normalized
warm start, and delegates all trainable PEPS tensors to Quimb TNOptimizer.
The global objective differentiates through its own boundary contractions;
it does not use the sweep optimizer's fixed FIT row/column environments.

- Default algorithm: NLopt `LD_VAR2`, maximum **1200 objective evaluations**.
  These are separate from sweep's LD_LBFGS/maxeval=50 defaults.
- Default autodiff backend/device: Torch/CPU, with the configured stabilized
  Torch SVD/QR registration when enabled.
- Default objective contraction: Quimb MPS, separate norm/overlap caps from
  `boundary_chi`; default `(4D,5D)`. Boundary sequence is
  xmax, xmin, ymin, ymax. Optional exact/CTMRG/hyper objective modes exist.
- Target norm is supplied as one for the default unitary workflow. Each
  objective measures the candidate norm and candidate-target overlap, not
  the target norm. Unknown target norms use the driver policy already tested.
- `cost_f="fid"` actually returns `abs(1-F)`, where
  `F=|<candidate|target>|^2/(norm_candidate*norm_target)`. With accurate
  contractions this agrees with infidelity; an invalid F>1 estimate is
  folded positive, unlike the sweep safeguard. This is an accuracy/diagnostic
  limitation, not a new proof of incorrect results under accurate boundaries.
- Default outer postcheck compares against the saved warm start. Tensor
  shapes/bond caps stay fixed during global parameter optimization.

## Confirmed findings

### High priority: Torch input fails at the outer overlap check

The installed Quimb `TNOptimizer.get_tn_opt()` converts returned variables
to NumPy. Neither GlobalOptimizer nor the PEPS wrapper restores the original
backend/device. With a Torch target still present, the postcheck mixes Torch
and NumPy arrays and raises:

`TypeError: tensordot(): argument 'other' (position 2) must be Tensor, not numpy.ndarray`.

Reproduced with real 3x3 complex128 Torch input, two RZZ gates, global
LD_VAR2/50 evaluations and otherwise valid contractions. Failure occurs in
the outer boundary overlap after optimization has completed. The corresponding
NumPy case passes. Independent native U1/U1U1 Torch global optimizations also
return NumPy blocks, confirming the conversion extends to native arrays.

Proposed: restore native array backend/device/dtype on the returned candidate
before delegated normalization and driver comparisons, using the owning
backend conversion helpers; regress dense Torch and native block output.

### Normalization ownership remains incorrect

The global wrapper always populates `normalize_kwargs`, so GlobalOptimizer
normalizes its output. The PEPS driver then normalizes it again with
`normalize_final=True`. With `normalize_final=False`, global normalization
still happens. This independently reproduces the earlier review finding.

Recording wrappers around real normalization calls showed:

- Default: three driver normalizations (initial input, warm start, candidate)
  plus one delegated candidate normalization.
- Opt-out: two driver normalizations (initial input, warm start) plus one
  delegated candidate normalization.

Proposed: make the PEPS driver own final normalization and honor its opt-out,
while defining how explicit delegated overrides interact with that policy.

### NLopt exception recovery does not restore the best iterate

`GlobalOptimizer.optimize_nlopt` catches NLopt exceptions and calls
`tnopt.get_tn_opt()` under a comment claiming best-state recovery. Installed
Quimb instead extracts the current vector, which can be the last bad trial.
It does not recover a best vector there. The inner catch also prevents the
PEPS wrapper's NLopt-to-LBFGS fallback from seeing those exceptions.

Controlled fault injection used a normalized 2x2 D=2 PEPS, seed 43, with an
identical target. The substituted NLopt loop evaluated the initial vector,
then added NumPy seed-71 Gaussian noise to the real packed vector, evaluated
that trial, and raised `nlopt.RoundoffLimited`. These were real objective
evaluations; only the failure/scheduling was injected. History was
`[6.66e-16, .987431843617645]`; the returned state's exact loss was
`.987431843617645`, proving best-state recovery does not occur.

This is not evidence that normal runs always return bad results. The driver's
default outer acceptance check can reject a worse candidate, but it cannot
recover a better intermediate state discarded inside global optimization.
Proposed: track and restore a valid best vector, report the early-stop status,
and avoid describing current-vector recovery as best-state recovery.

### Lower-priority standalone option inconsistency

`GlobalOptimizer.normalize()` uses `self.norm_kwargs`, ignoring separately
configured `self.normalize_kwargs`. The optimization entry points do use the
latter. For example, constructor norm mode exact and normalization mode MPS/
chi=1 result in public `normalize()` using exact/chi=64. Confirmed by code
inspection; this does not cause the default PEPS wrapper's normalization bug.

## Real numerical evidence

Primary fixture: normalized random 3x3 D=2 complex128 PEPS, seed 311;
RZZ(.37) on ((0,0),(0,1)) and RZZ(.23) on ((1,1),(2,1)); one automatic
batch; greedy contraction optimizer. Exact reference from separately gated
dense vectors. Original input and retained bond cap checked.

| NumPy run | Pre-infidelity | Exact final infidelity | Evaluations |
| --- | ---: | ---: | ---: |
| Global defaults, including cap pairs (8,10) | .1206935569586185 | .04920040177648155 | 682 of 1200 |
| LD_VAR2, boundary (16,24), norm/evaluation (32,48) | .12069355695861783 | .0632327432579306 | 50 |
| LD_LBFGS, same larger caps | .12069355695861783 | .06224135324268942 | 50 |

The full-default run took about 19.8 seconds in this CPU probe, preserved
the input, retained D=2, and had dense norm error 6.7e-16. Reported final
infidelity matched exact contraction within 2.3e-15. An additional 1200-budget
larger-cap run converged in 686 evaluations to .049200401114201764.
These individual timings/results do not establish an algorithm ranking.

The normalization opt-out probe (LD_VAR2/n=2) and Adam/n=1 with outer final
measurement disabled also completed on NumPy. For these successful runs,
loss-history endpoints matched the returned-state exact loss; no stale-loss
bug was reproduced on those paths.

Independent objective/gradient probe: random normalized 3x3 D=2 Torch PEPSs,
seeds 291/293; complex Gaussian direction seed 301 scaled by .02; centered
finite differences with step 1e-5. Exact, MPS cap 32, and CTMRG cap 32 losses
matched dense fidelity to displayed precision. Maximum directional gradient
error was 3.53e-12. MPS cap 2 had expected approximation error 2.57e-5 and
gradient error 1.74e-12 against finite differences of that same approximate
objective. This is one small CTMRG fixture, not broad CTMRG validation.

Native standalone global checks: fermionic U1 and U1U1 2x2 states, seeds
7/107, bond dimension 2, physical dimensions 2/4, alternating U1U1 site
charges, explicit Torch complex128 blocks. LD_LBFGS/n=5 reduced each loss
from .9575355990455918 to .050934773841351366. Returned blocks were finite
but converted to NumPy, as reported above. No native PEPS-driver end-to-end
success is claimed from these standalone runs.

## Validation and scope

Fresh selection: **193 passed**, 44 warnings, 23.65 seconds across
`tests/test_optimize_global.py`, `tests/test_optimize_peps.py`, and
`tests/test_peps_optimizer_batching.py`. Passing tests do not cover the
Torch-driver failure found by the independent real probe.

Review-note links and `git diff --check` passed. No production/test changes,
full suite, GPU, broad complex64, hyper-compressed objective, or multi-batch
backend success claim. Added this note and its history entry only; no commits
or pushes. Findings above are **unfixed** in this review.
