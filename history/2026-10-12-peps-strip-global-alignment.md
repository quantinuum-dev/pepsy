# PEPS strip batches and optimizer alignment

- Branch: `develop`; baseline `192561a`, already ahead of origin by two commits.
- Status: working-tree changes only; nothing staged, committed, or published.
- Scope: requested fixed column/row gate targets, SU guesses, reusable boundary
  MPS, independent norm/overlap caps, and matching global/sweep target-norm and
  backend behavior. FullUpdate plots were refreshed earlier in this session;
  no new production trajectories were launched.

## Changes

See the [API recipes](../docs/api/optimizers/peps.md) and
[implementation/audit note](../docs/development/notes/2026-10-12-peps-strip-global-alignment.md).
Global fitting supports one/two-strip and full-queue targets, Quimb boundary
methods and layer order, and Torch/JAX registration before differentiation.
ALS `auto` uses a positive-support spectral solve for small matrices and CG
for large ones. Boundary reuse transfers only compatible guesses and refits
them; exact targets and numerical cut caches remain separate.

## Validation before the lazy-target extension

- Before the shared dependency change: 506 focused sweep/boundary/batching
  checks passed; a later regression selection passed 589 tests with one
  package-layout subprocess exiting on SIGBUS (cause unconfirmed).
- New global direct-boundary batch cases passed finite-difference gradients
  and decreased fidelity loss on Torch/JAX for one/two-strip/full-queue targets.
- The first broad suite was interrupted after 392 passes and two skips once
  the shared dependency replacement was identified. No full-suite pass is
  claimed. Subsequent mixed-backend failures in that environment are recorded
  in the audit note rather than repaired by editing installed libraries.
- Isolated default smoke: 94 passed. Isolated package-layout and fermionic
  boundary checks: 22 passed, including the previously crashing subprocess.
- Native Torch/JAX batched-global and standalone-gradient cases: 9 passed.
- Final combined domain selection: **680 passed**, covering sweep solvers,
  batching, boundary engines, global fitting, backend rules, and public APIs.
- Additional driver, sweep-performance, and boundary-convergence selection:
  **207 passed**.
- Isolated extended-suite attempt: 582 passed, two skipped, then manually
  stopped in unrelated cluster coverage after five minutes. No failures had
  been reported; this is not a complete extended-suite result.

The isolated interpreter is `/tmp/pepsy-strip-validation-env/bin/python`.
Its site configuration loads `/tmp/pepsy-strip-validation-deps` before the
shared dependency path, including in Python `-I` subprocesses. Activate it
before reproducing tests. Ruff and both repositories' diff checks pass.

## Lazy-target extension

Strip ALS now retains tagged split-gate factors, including all factors in its
matrix-free RHS. Global lazy targets remain opt-in. Constant diagonal gates
use their exact rank bound on concrete JAX arrays too; traced values retain
all channels. `canonize` is exposed for direct global boundary contractions.

- Domain regression after this extension: **267 passed**, including strip
  ALS (spectral/CG), boundary FIT with diagonal/general unitary gates, existing
  ALS and batching, global optimizer behavior and safeguards.
- Torch/JAX direct-boundary global batch and standalone gradient tests:
  **16 passed**, including exact target reconstruction, directional finite
  differences and independently evaluated fidelity improvement.
- Concrete-JAX target reconstruction plus safe tracing of rank detection:
  **1 passed**.
- Additional general full-rank two-qubit gates in lazy global fits:
  **6 passed** across Torch/JAX and one/two-strip/full-queue batches, using
  the usual canonicalization settings and the same gradient/fidelity checks.
- Small storage/runtime measurements and initially failing numerical cases
  are documented in the audit note. They do not demonstrate faster repeated
  contractions or lower peak workspace.

The new paper [Burgelman et al., arXiv:2607.15030](https://arxiv.org/pdf/2607.15030)
was assessed for implicit environment differentiation. It is a research route
for global fitting through a converged boundary, not implemented in this
patch. Finite boundary-FIT stationarity equations and gauge handling would
need separate derivation and validation; current one-site ALS needs no AD.

## Follow-up: default sweep policy and axis traversal

The user requested that `mode="sweep"` select the new settings, changed
default caps to `(2*D**2, 4*D**2)`, and clarified the two nested sweep budgets.
The driver now starts at the gated strip, continues to the final parallel
strip, then completes two backward/forward rounds over the axis. Each strip
visit permits two inner ALS round trips. Gate batching, SU guesses, lazy
targets, spectral/CG selection, fixed caps, and `reuse-direct` initialization
are defaults. Explicit settings and global/full-update dispatch are retained.

`SweepOptimizer.optimize_axis(start_index=...)` supplies the traversal;
`optimize_strip` still optimizes only one strip. Scalar step records now expose
per-visit infidelity, direction, strip index, and local cache counters.
Tests of the previous auto-batching/gradient paths specify those options
explicitly; expected automatic caps were updated to the requested formulas.

- New defaults, run-time mode overrides, and middle-strip cache correctness:
  **8 passed**. Each middle-strip visit was compared with an independent dense
  fidelity contraction, including both return directions and both axes.
- Strip, expression-cache, and single-strip performance selection: **74 passed**.
- Combined driver/defaults/batching/full-update/adaptive-boundary selection:
  **373 passed**.
- Default smoke profile: **94 passed**.
- Additional global Torch/JAX, ALS solver, smart-order, and public/package
  checks: **260 passed, 2 failed**. Both failures exposed an unnecessary
  layered-target restriction on pure one-site queues with custom gate
  transforms. The guard now applies only when two-site gates are present.
- After that fix, smart-order/defaults/strip/driver/batching/full-update
  regressions: **357 passed**, including both previously failing cases.
- Final Ruff check of `src tests` and both repositories' `git diff --check`
  passed. These selections overlap and are not a full extended-suite result.

The cache audit reused the existing moving-boundary and local ALS cursor
implementations. Updated strips are absorbed before subsequent solves; local
prefix/suffix environments rebuild on direction reversal, and reusable
contraction expressions receive fresh tensor arrays. No separate cache
algorithm or production benchmark was added. All changes remain uncommitted.

## Remaining limits

The shared environment is outside this patch and changed while testing.
Final checks use a compatible isolated dependency snapshot. New long-run
fidelity/runtime comparisons remain unmeasured; historical FullUpdate curves
must not be relabeled as strip-batch results. Native ALS backend limits remain
NumPy/Torch/CuPy; this change adds no JAX or Symmray ALS implementation.
