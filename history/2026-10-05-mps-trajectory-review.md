# 2026-10-05 — MpsOptimizer trajectory correctness and efficiency review

- Scope: review trajectories; user trusts pure-state unitary replay. No runtime
  implementation or test edits made for this review.
- Branch / baseline commit: develop / d7d509b.
- Commit status: uncommitted journal. Earlier stabilizer compression changes
  remain in the working tree; they were preserved, not included as new fixes.
- Environment: local genpy Python 3.12, existing installed dependencies.

## Confirmed correctness findings

1. Importance proposals can sample a physically impossible Kraus outcome.
   `ImportanceSamplingPolicy.probabilities` permits positive proposal mass on
   zero target mass. `_apply_trajectory_event` and its coalesced counterpart
   then apply that zero-norm branch and try to normalize it. Reproduction:
   start in `|0>`, amplitude damping gamma=.2 on site 0, proposal
   `{'jump': .5, 'no_jump': .5}`, eight shots, seed 6, mode direct.
   Both independent and coalesced strategies raise FloatingPointError about
   a zero canonical-center norm. This accepted proposal should either receive
   an explicit validation error before branch application or have defined
   zero-weight handling that never normalizes a null state.

2. Importance support validation ignores positive targets <=1e-14.
   `noise.py:301` uses `(target > 1e-14) & (proposal <= 1e-14)`.
   A target `[1-1e-16, 1e-16]` with proposal `[1, 0]` is accepted. This cannot
   produce an unbiased estimate of the rare branch, contrary to the public
   policy contract. Check strictly positive target support against strictly
   positive proposal support, keeping distribution roundoff checks separate.

3. State-dependent proposal callbacks lose the optimizer in coalesced mixtures.
   The mixture path at `noise.py:5538` computes one proposal for every live
   parent without passing an optimizer. Independent mixtures and coalesced
   Kraus events do pass the live parent. A callback that checks its optimizer
   argument and returns target probabilities succeeds independently but fails
   coalesced because optimizer is None. The documented callable contract is
   `(event_index, labels, target_probabilities, optimizer)`. Evaluate callbacks
   per parent for state-dependent mixture proposals, reserving other live
   parents in branch-cap accounting just as the Kraus path does.

## Efficiency findings

- `_kraus_probabilities` reads the initial norm, then every successful MPS local
  outcome path re-reads it in `_mps_outcome_norm_squared`. Instrumentation found
  three `_trajectory_norm_squared` calls for one two-outcome amplitude damping
  event. Reuse the base norm or return normalized local probabilities directly.
  Canonical reads are local, but each scalar conversion can synchronize a
  device. No GPU timing or performance claim was established here.
- Independent replay appends one diagnostic snapshot dictionary per shot at
  `noise.py:5249`, even with retain='none', and even when retained optimizers
  will be used instead. Fold the maximum residual and fallback flag online.
  This removes O(shots) diagnostic storage; existing MPS retention already
  follows the requested policy.
- Coalescing demonstrably shares deterministic prefixes and suffixes. An
  eight-site product circuit, eight H gates before and after one .02 bit-flip
  mixture, 64 shots, seed 31, direct mode used one factory invocation and 26
  replayed actions across two retained leaves. Independent replay used 64
  factory invocations and 1088 actions across 64 states. Single CPU timings
  were .0127s versus .1483s; these are a smoke measurement, not a general
  benchmark or a forecast for entangled large-MPS simulations.

## Validation and positive evidence

- Focused pytest: test_trajectory_noise.py, test_mps_controls.py,
  test_mps_dynamic_controls.py: 214 passed, two failed. test_mpi.py: 56 passed,
  one failed. All three failures request removed mode='dmrg1'; they are stale
  test expectations, not observed trajectory numerical failures. No test
  expectations were changed to hide these failures. No real multi-rank MPI
  integration or full repository suite run.
- Independent dense audit on a three-qubit circuit: H, a nonlocal CNOT,
  amplitude damping, a seeded complex two-site QR unitary with reversed
  support order, then two more amplitude damping events. Replay each sampled
  record independently using dense tensor axis permutations and explicit
  Kraus normalization, and compare its probability and final state.
- Modes: direct, dmrg2, dmrg3, svd, swap, perm, exact, exact-batch; independent
  and coalesced strategies, 16 shots each, seed 27, chi=8, cutoff=0, n_iter=4.
  All 160 retained states/branches matched: maximum probability error
  3.33e-16, absolute infidelity 1.11e-15, norm error 4.44e-16. This verifies
  the tested full-rank paths, not accuracy under aggressive truncation.
- Existing focused tests cover probabilities before truncation, physical
  branch normalization, entangled reset/cap/leakage, persistent layouts,
  pair-budget/control forwarding, seed consistency, copy counts and terminal
  sampling reuse. Rare positive amplitude damping branches of probability
  1e-40 pass direct/dmrg2 tests for independent and coalesced MPS trajectories.
- Local probabilities are evaluated before selected-branch compression;
  nonunitary replay disables unitary norm restoration, selected Kraus states
  are normalized, and physical branch exponents are cleared.
- git diff --check passed before and after writing this journal. No runtime
  implementation change, commit or push.

## Proposed next step

Fix the three importance-sampling contracts with focused regression tests,
then remove redundant norm reads and diagnostic snapshot retention. Update
the three stale dmrg1 parametrizations while preserving explicit tests that
the removed mode is rejected. These proposals have not been implemented by
this review.

## Autoray documentation follow-up

At the user's request, read the linked Autoray latest documentation, the
[automatic dispatch guide](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html),
API reference and official main-branch implementation. The installed Autoray
0.11.1.dev3+g1b476b305 exposes get_namespace, from_numpy, to, and to_device.
The documentation pages showed inconsistent version banners, so no release
upgrade or version equivalence is inferred from the `latest` URL.

Relevant guidance: bind array creation to a live state/environment array with
get_namespace(like=array), or explicit do(..., like=array). Keep contractions
and normalization on that backend. A backend-name string alone does not
specify the state device/dtype. Autoray dispatch does not automatically convert
all existing operands; retain Pepsy's explicit generated-gate conversion and
user-payload validation. Avoid a shared global backend context during threaded
shot execution. Rebuild state-bound caches when backend/device/dtype changes.
Keep scalar probability readout and host RNG separate from tensor operations.

CPU smoke checks: Torch and JAX, direct/dmrg2/exact, two-site |10> input,
amplitude damping .2, four coalesced shots, seed 6. Every retained tensor stayed
on the initial backend and two branches survived in all six combinations.
No GPU availability or device-transfer performance was established.

Additional confirmed boundary issue: the one-site exact MPS local Gram helper
at noise.py:3042-3043 calls np.asarray on both the dense state and Gram matrix.
Instrumentation observed shape (2,1) and (2,2) backend arrays entering NumPy
twice per two-outcome channel, on both CPU Torch and JAX. This bypasses backend
contraction and can require host transfer or fail for device arrays. Replace
that local Gram expression with backend reshape/conjugate/matmul/reduction and
convert only its final scalar. This finding is recorded, not implemented.

## Implementation after the user's explicit request to fix all findings

Implemented the three importance-sampling corrections in noise.py:

- Validate strictly positive target/proposal support with no rare-event
  threshold. Reject proposal mass on zero target probabilities before branch
  application with a descriptive ValueError; a null physical state is never
  normalized. This intentionally requires equal target/proposal support.
- Callable coalesced mixture proposals now run per parent with that parent's
  optimizer, using the existing remaining-live-parent branch budget. Fixed
  mixture proposals retain their shared distribution path.
- One-site exact local Gram evaluation uses Autoray reshape, conjugate,
  matmul and reduction on existing backend arrays, reading only scalar
  values. Adopted Autoray's array-based dispatch; no dependency upgrades or
  installed-library edits. Reused the unchanged session's upstream audit.

MPS local outcome probabilities reuse the already computed base norm. A
two-outcome canonical channel now calls _trajectory_norm_squared once,
instead of three times. Discarded-state diagnostics fold online into one
summary for both ordinary independent and deferred magic replay; retained
states no longer generate unused per-shot snapshots.

Added test_trajectory_importance_regressions.py, updated the noise API guide
and changelog. Removed dmrg1 only from the three stale success parametrizations
in trajectory/MPI tests; existing constructor/set_mode rejection tests remain
and pass. Earlier stabilizer working-tree changes are preserved.

Fresh validation:

- Trajectory regressions, existing trajectory suite, MPS controls/dynamic
  controls, MPI unit tests, MPS backend tests and removed-mode rejection tests:
  318 passed, 19 skipped. All skips concern unavailable CUDA, CuPy or Metal.
- Final expanded regression file with two logical JAX CPU devices:
  13 passed, two unavailable CUDA/CuPy skips. Adds deferred-injection retention
  checks and explicit nondefault JAX device placement. Combined selections
  cover 321 distinct passing cases and 19 remaining hardware/dependency skips.
- Regressions verify impossible outcomes reject before replay, positive target
  mass of 1e-40 cannot be omitted, tiny positive proposal mass is accepted,
  callbacks observe both prebranch parent states, recorded ratios match the
  parent-specific proposals, branch caps reserve other parents, one base norm
  read is shared, and diagnostics survive aggregation without per-shot storage.
- Torch/JAX one-site exact tests prohibit non-scalar backend arrays from
  entering np.asarray, compare amplitude damping probabilities to [.7, .3],
  require the local path without copied-state fallback, and check backend,
  dtype and device preservation. The second JAX device is a CPU device, not
  GPU validation.
- python -m ruff check src tests and git diff --check passed. No full repository
  suite or real multi-rank MPI run; no commit or push.
