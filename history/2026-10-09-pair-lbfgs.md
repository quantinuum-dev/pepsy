# 2026-10-09 — Optional full-tensor L-BFGS for PEPS pairs

- Scope: keep gauge conditioning optional/off, retain reduced tensors by
  default, add joint full-tensor L-BFGS, and review the two supplied papers.
  Interpreted the reduced/gauged pair request as `PepsOptimizer` two-site full
  update; `SimpleUpdateGen` was not changed.
- Branch / baseline: `develop`, `21f883b`.
- Commit status: working-tree changes only; nothing staged, committed or
  published. Preserved pre-existing full-site layer/strip and MPS/JAX edits.

## Changes and findings

- Explicit `tensor_mode="reduced"` default; `gauge=False` was already in the
  working tree. `tensor_mode="full"` selects joint L-BFGS. Reduced mode also
  accepts explicit L-BFGS. Full mode changes all entries of both active sites.
- Reused pair environment, positive projection and acceptance logic; added
  analytic gradients and a dense full-environment size guard.
- Updated API docs, module map and changelog. The supplied papers distinguish
  reduction, positivity and conditioning; their algorithms do not imply that
  L-BFGS or a full-tensor fit requires gauge conditioning.
- Numerical details, paper differences, upstream audit and limitations are
  in the [evidence note](../docs/development/notes/2026-10-09-pair-lbfgs.md).

## Validation and limits

- New pair L-BFGS plus FU/strip/layer/API/layout suites: **160 passed**,
  three warnings, 30.61 seconds. Includes actual CuPy CUDA execution.
- Ruff `src tests` and `git diff --check` passed. No fresh full-suite run.
- Full mode is bounded/dense and large-D performance is unverified;
  `full_max_matrix_size=1024` requires an override for bulk D=4.
- The second paper's plaquette/second-neighbor update and its different
  spectral positive approximant were reviewed but not added.

## Follow-up — Quimb availability and scaling review

The user requested checking existing Quimb support and D/chi scaling before
judging the implementation. Found its dedicated
`gate_full_update_autodiff_fidelity` and `FullUpdate` strategy, beyond the
generic `TNOptimizer` previously mentioned. The earlier upstream assessment
was incomplete. A custom frozen-environment TNOptimizer prototype matched
raw dense values/gradients to about 1e-15 and lowered loss in four small CPU
cases; eager and traced paths were exercised. The dedicated wrapper needs
complex-conjugate-view/native-output care in this installed version.

Dimension-only Cotengra path probes demonstrate D/chi-dependent scalar-TN
scaling without allocating large full metrics. Small cached dense kernels
were faster in the CPU microchecks, so no blanket timing claim is justified.
The full dense PSD solve scales as D^18 and is unsuitable as the large-D
design. See the [audit](../docs/development/notes/2026-10-09-quimb-pair-autodiff-audit.md)
for assumptions, topology, measurements, objective differences and recommended
reuse. This follow-up changed only notes; production integration was not
performed. Existing uncommitted implementation edits remain intact.

## Follow-up — Cached scalar objectives and full-update defaults

The user subsequently authorized production integration: only reduced pairs
form norm matrices; full pairs use cached scalar TN contractions and autodiff,
with the gate retained explicitly. Implemented this using the existing shared
`GradientOptimizer` and public Cotengra constant-folded expressions. Removed
the dense full-pair route and its size guard. Reduced ALS remains default,
gauges stay off, and the user's final boundary policy is one `chi=2*D**2`,
cached DMRG MPS, no convergence probes or automatic evaluation cap retries.
Explicit opt-ins remain. Same-cap boundary retuning now preserves canonicalized
lower-rank cuts rather than repadding and invalidating their cache entries.

The user's next refinement adds an untruncated-SVD shortcut: local SVD
initialization whose numerical rank fits Dcap is accepted with local fidelity
one, without FU environment construction or optimization. The rank test uses
machine precision, independently of fit tolerance. Requested normalization
remains. Tests exercise product-state growth, equality with the cap, both
tensor modes, complex64/128, and a small real truncation that must still run FU.
The user explicitly confirmed local SVD warm start as the intended
initialization, without persistent SU bond weights.

See the [implementation note](../docs/development/notes/2026-10-09-cached-pair-tn.md)
for semantics, cache ownership, backend handling and cost limits. All changes
remain uncommitted on the same branch; pre-existing edits were preserved.

Validation after the exact-SVD shortcut: **554 passed**, seven warnings,
132.67 seconds across pair/exact-warmstart, FU Torch/CuPy, strip/layer,
PEPS driver/convergence, shared boundary, public API and layout tests.
Ruff `src tests` and `git diff --check` passed. A final narrowing of the retuning
guard checks geometric capacity, so genuinely compressed guesses can still
regrow. Its new regression plus boundary/convergence/FU/exact-warmstart suites:
**302 passed**, two warnings, 71.94 seconds. Ruff and diff checks also passed.
The full run started before this final guard refinement: **8,316 passed,
9 skipped, 2,088 warnings**, 2,931.79 seconds. Command: activated Python with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg`,
then `python -m pytest -q -o addopts=''`. Log:
`/tmp/pepsy-pair-full-suite.log`. This run used the preceding cache guard,
supplemented by the final 302-test recheck; it is not a second full run after
that small refinement. Concurrent unrelated FIT sum-target edits appeared by
the final status check and were left untouched. A final current-tree check of
the defaults/cache, rank-growth safeguard, exact warm starts and scalar TN
objectives passed **15 tests** in 5.55 seconds; Ruff and diff checks passed.
This was a mutable working tree, not validation of a frozen commit. No failures
remain in the executed checks. Changes are still uncommitted.

## Follow-up — Gate-order and two-level cache audit

Checked the user's requested alignment between compiled gate traversal,
transverse boundary MPS reuse, and within-row/column environment reuse.
`ordered_gate_run` orders/fuses the queue before FU visits it. Boundary cuts
remain cached across strips; exact prefix/suffix entries retain only the
active strip and reset at a turn. Both validate tensor dependencies. The
default input order is unchanged; row/column/smart traversal can improve
inner reuse when reordering is legal. No production fix was needed.

Added [ordered-cache regressions](../tests/test_peps_ordered_environment_cache.py)
for all four policies with reduced ALS and full-tensor L-BFGS. They check
executed strip coordinates, boundary reuse, inner hits in both directions,
smart local-gate fusion, and agreement with fresh environments under the
same ordering. An ample boundary cap isolates cache correctness from
approximate boundary fitting. Documented the cache scope and ordering
semantics in the [PEPS API guide](../docs/api/optimizers/peps.md).

Current checks: **56 passed**, one warning, 26.58 seconds for gate ordering,
smart compilation, ordered caches, scalar objectives and existing local-change/
corner-turn cache regressions. CuPy in-place invalidation and both-axis cache
checks: **2 passed**, 4.03 seconds. Ruff `src tests` and `git diff --check`
passed. No full-suite rerun for this tests/documentation-only follow-up.
Concurrent unrelated edits were preserved; nothing committed or published.

## Follow-up — Boundary modes, CuPy ALS, and D/chi costs

The user requested verification of direct/DMRG2 boundary modes, CuPy at all
levels including ALS, and the proposed D^8 total scaling. Added mode/dtype/
ALS-solver combinations to the existing CuPy regression, including both
host-transfer guards and native dtype/device checks. **28 tests passed**, one
warning, 9.83 seconds, including existing cache/adaptive checks and the full
L-BFGS CuPy/Torch comparison. No production behavior changed.

The [cost audit](../docs/development/notes/2026-10-09-fu-boundary-costs.md)
records source inspection and dimension-only Cotengra paths. The standard
double-layer one-site boundary/pair contractions cost chi^3 D^4 + chi^2 D^6,
or D^10 at chi ~ D^2; representative peak intermediates instead scale as
D^8. Reduced dense ALS costs D^6. Dense two-site boundary SVDs cost D^12;
standard joint direct canonicalization can reach D^14 from expanded bonds.
These are bulk arithmetic estimates with explicit rank/path assumptions,
not measured timings or guarantees for every heuristic path. The API guide
now explains modes, CuPy/host distinctions and links the detailed audit.
Ruff and diff checks passed; no full-suite rerun, commit, or publication.

## Follow-up — Sequential direct defaults and joint DMRG targets

The user clarified that direct compressors should absorb/compress BRA and
KET separately, while DMRG/DMRG2 should fit a joint uncontracted target.
Implemented this default selection in `PepsOptimizer` and propagated it
through its existing boundary policy; standalone boundary helper defaults
are unchanged. Explicit layer settings take precedence. SRC initial guesses
for variational boundary fitting remain the existing default.

Extended the [cost audit](../docs/development/notes/2026-10-09-fu-boundary-costs.md)
with actual QR/SVD matrix shapes: sequential direct has expanded bond chi D
and grouped local dimension O(D^2), giving D^11 canonicalization at chi=D^2;
the earlier D^14 estimate applies only to joint direct. Production two-site
FIT splits confirmed square chi D^2 matrices. Reduced norm conditioning is
Hermitianization O(D^4), PSD eigendecomposition/reconstruction O(D^6), and
O(D^4) matrix storage for fixed physical dimension, independent of chi after
construction. Gauge fixing remains a separate option.

Added policy forwarding, explicit override, and actual separate-layer/joint
FIT target-array checks, plus CuPy zipup and both explicit absorption orders.
An initial new test expected presentation layer tags to survive FIT's existing
retagging; corrected it to inspect preserved original array identities and
unfused graph structure. The new targeted recheck passed eight tests.
All edits remain uncommitted; unrelated concurrent edits were preserved.

The broader check exposed automatic layer-policy leakage into exact metrics
and the separate native Quimb MPS engine. Resolved the policy for the actual
metric route while preserving explicit settings, and updated the conservative
initial-loss reuse comparison to recognize the new direct default. The metric/
driver/safeguard recheck passed 233 tests, with one remaining initial-loss
case subsequently included in the final FU recheck. These findings did not
require changing the boundary or FIT contraction kernels.

Final FU/cache/ALS/L-BFGS/CuPy/exact-warmstart check, including the corrected
initial-loss case: **130 passed**, one warning, 36.54 seconds. Together with
the 233 passing metric/driver/safeguard cases this covers all **363** selected
cases, with no outstanding failures. Logs:
`/tmp/pepsy-layer-metric-recheck.log` and `/tmp/pepsy-layer-fu-recheck.log`.
Ruff `src tests` and `git diff --check` pass. No full-suite rerun for this
optimizer policy change; no commit or publication.

## Publication and final full-update engine defaults

The user authorized committing and pushing this task to `develop`. Staging
separates the pair/FU/default/cache work from pre-existing layer refinement,
MPS/JAX, and concurrent FIT sum-target changes, including mixed-file hunks.
An isolated checkout of the first staged snapshot passed **626 tests**, seven
warnings, 91.46 seconds, plus Ruff and relative-documentation-link checks.
The numerical checks imported this isolated checkout rather than the shared
working tree. The fetched remote remained at baseline `21f883b`.

Before publication, the user clarified that FU should own the complete gate
sequence and use local/accumulated fidelity without outer global pre/post
checks by default. Added mode-dependent measurement and acceptance defaults,
preserving explicit opt-ins and other modes' behavior. Added the explicit
`accumulated_local_fidelity` product field. Tests forbid independent checks,
verify constructor/run aliases, exercise reduced/full solvers and trailing
single-site gates, and retain explicitly enabled acceptance/retry tests.
Requested normalization and the solvers' local best-candidate handling remain.

The final isolated staged code passed **628 tests**, seven warnings, 90.21
seconds, covering PEPS driver/FU, CuPy, boundary convergence, exact warm starts,
pair objectives/L-BFGS, ordered caches, safeguards, existing strip refinement,
shared boundary preparation, public API and package layout. Ruff and staged
diff checks passed. Log: `/tmp/pepsy-pair-staged-final.log`. This final selection
validates the publication snapshot without relying on unrelated workspace
edits; it is not a fresh full-suite run. This entry accompanies the user's
authorized commit/push; the commit identifier and push outcome are reported
in the session handoff.
