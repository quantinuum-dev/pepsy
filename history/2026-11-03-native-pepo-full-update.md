# 2026-11-03 — Native PEPO two-site full update

- Scope: extend the optimized `PepsOptimizer` full-update gate stream to PEPO
  inputs, retaining QR/LQ reduction, positive norm environments and ALS.
- Branch / baseline: `develop` / `223504e`.
- Publication: user authorized committing this change set and pushing
  `origin/develop` after validation. Git history and remote refs carry the
  resulting publication record; the sections below retain earlier status.

## Implementation

`ReducedPair` uses the existing PEPO-aware physical-leg reduction. It contracts
the gate directly into the selected upper/lower reduced leg, retaining the
other leg as a batch axis rather than constructing a larger superoperator.
Reconstruction restores both native operator legs. PEPS calculations retain
their original single-physical-leg contraction.

The driver accepts default upper legs, constructor `which`, and per-entry
overrides. One-site actions and exact refinement targets use live PEPO index
names. Lower-leg application follows the existing physical-index gate
convention, $O G^T$; supplying `G.T` gives right multiplication by $G$.
No new gate, QR, PSD, ALS or boundary kernel was introduced.

The full-tensor pair objective retains both physical legs and applies its gate
lazily. Pair L-BFGS, strip/layer refinement, exact-SVD shortcuts, normalization
and checked boundary/strip caches work with PEPOs. Refinement's local matrix
size excludes both physical legs. Reports identify the chosen leg and the
normalized Frobenius operator-overlap convention. PSD projection acts on the
environment, not the evolved operator.

Public usage and operator-amplitude preservation are documented in the
[PEPS/PEPO guide](../docs/api/optimizers/peps.md#native-pepo-full-update).
`normalize_initial=False` and `normalize_final=False` retain operator amplitudes;
ordinary normalization instead sets the Frobenius norm to one. JAX,
native symmetry and differentiation through full update remain unsupported.

## Dependency audit

Environment: shared Python 3.12; Pepsy 0.6.0, Quimb
1.15.1.dev101+g1b64353d9, Autoray 0.11.1.dev14+g014a3f69a, Cotengra 0.8.3,
Cotengrust 0.2.1, Symmray 0.4.1.dev25+gdc377f033, Torch 2.14.1+cu126.
Inspected installed gate-index application and Tensor fuse/unfuse/view APIs;
reviewed the official Quimb/Cotengra changelogs and Autoray/Symmray repositories.
The Symmray abelian-array documentation URL was unavailable; its repository
and installed implementation remain available. Classification: **adopt** existing
public Quimb tensor/gate operations; no compatibility shim or dependency change.

## Validation

- New native PEPO tests: **33 passed**, including actual Torch CUDA and CuPy
  complex64/128 runs. Dense operator actions independently check both legs,
  reversed pairs, mixed one-/two-site streams, entry overrides, exact-SVD
  shortcuts, custom leg metadata, operator scale, explicit one-site physical
  indices and unequal physical dimensions.
- ALS/Quimb, weighted QR, reduced/full pair L-BFGS, positive projection and
  singular environment gauges, normalization, refinement and cache reuse are
  checked. Cached and uncached PEPO updates agree; reported local fidelity
  agrees with dense Frobenius fidelity on exact small environments.
- Full-update/gauge/pair/refinement selection: **169 passed**.
  `/tmp/pepo-fu-regressions.log`.
- Optimizer/whole-PEPO-step/CuPy selection: **253 passed**.
  `/tmp/pepo-fu-integration.log`. These selections overlap the new tests;
  their counts must not be added as distinct tests.
- Default smoke profile: **112 passed**, `/tmp/pepo-fu-smoke.log`.
- Ruff on the changed Python files, documentation link checks and
  `git diff --check` pass.
- New PEPO tests: `/tmp/pepo-fu-native-final.log`. Existing Quimb split-default,
  upstream NumPy deprecation and rank-deficient QR warnings remain.

The complete extended suite and production-scale performance were not measured.
The numerical checks used spare CPU cores with one BLAS/OpenMP thread.

## Second careful review

The review found one test-integration gap: the new module was absent from
the central filename-to-domain map and had no module-level domain marks.
It now declares `peps` and `optional`, so domain/optional test commands select
its cases. No additional numerical implementation fault was found in the
tested paths, and no production kernel was changed during this review.

Expanded the native regression matrix from 33 to **51 tests**. New checks cover
actual pair fits with a nonzero network exponent and complex gate scale,
with and without target normalization; identity-operator evolution against
the independently optimal operator-Schmidt truncation; smart scheduling and
one-site fusion; all constructor/run-time full-update aliases; adaptive
boundary convergence; and invalidation after nonunitary in-place changes to
an exterior site's arrays. Unequal upper/lower dimensions now exercise ALS
and dense fidelity, beyond target reconstruction alone.

A 3×3 periodic PEPO interior pair agrees with dense Frobenius fidelity to
floating-point precision. Periodic wrap bonds still fail the existing
coordinate-adjacency check; this is tested and explicitly documented, not
claimed as new support.

The expanded native/full-update/gauge/pair/refinement selection passes
**198 tests**, including the 51 native cases and actual Torch CUDA/CuPy
checks. Log: `/tmp/pepo-fu-careful-review.log`. Earlier 112 smoke and 253
integration checks remain relevant to the unchanged implementation. These
test selections overlap. The corrected `-m peps` selection runs all **51 native
tests** successfully (`/tmp/pepo-fu-review-domain.log`); Ruff and whitespace
checks pass. The full extended suite and large-bond performance
remain unmeasured; changes are still local and uncommitted on `develop`.

## Arbitrary operator norm audit

The user's norm review identified a real driver assumption: unitary runs
with final normalization enabled could pass target squared norm one into
the pre-fit diagnostic, even for an arbitrary PEPO. PEPO diagnostics now
measure the target norm unless explicitly supplied. No assumed $2^L$ norm
was introduced: this value holds only for an unscaled qubit Pauli string.

Constructor `normalize_initial=None` and `run(normalize_final=None)` now
select the policy from the input: PEPS keeps its previous normalization
defaults, whereas PEPO initial/output normalization is off. Explicit booleans
retain precedence. `set_state` follows the new state kind only when the initial
policy remains automatic. Explicit nonunitary target normalization remains
available under the existing `normalize_target`/`non_unitary` policy.
Compression retains the least-squares amplitude rather than forcing either
unit norm or the source norm. Updated API docs, docstrings and changelog.

Current validation after these implementation changes:

- **82 native PEPO tests passed**, `/tmp/pepo-fu-norm-tests.log`. Scaled Pauli
  strings and operators containing a projector agree with independently
  optimal operator-Schmidt truncations on both legs, with both direct/DMRG
  boundaries, and with or without explicit output normalization. Their
  measured target squared norms are checked independently from dense arrays.
- Scaled entangled 2×3 operators agree with dense normalized Frobenius
  fidelity. Instrumentation verifies that the requested direct compressor
  or variational boundary solver actually executes. At boundary cap two,
  3×3 outputs remain homogeneous in an exterior operator scale for both
  methods, with scale-independent normalized local fidelity.
- Actual Torch CUDA/CuPy checks now include both direct and DMRG boundary
  modes, complex64/128 and upper/lower legs, retaining backend/device/dtype
  without bulk host conversion during the fit.
- **370 existing optimizer/full-update/pair/gauge/refinement/whole-PEPO-step
  tests passed**, `/tmp/pepo-fu-norm-regressions.log`.
- **112 smoke tests passed**, `/tmp/pepo-fu-norm-smoke.log`.
- **55 public API/package-layout tests passed**,
  `/tmp/pepo-fu-norm-api.log`; this selection overlaps smoke coverage.
- `ruff check src tests`, local Markdown target checks and `git diff --check`
  pass. Known upstream split-default/NumPy and existing rank-deficient QR
  warnings remain. The complete extended suite and production-scale
  performance remain unmeasured.

The first expanded-test attempts exposed two test setup errors, both fixed:
changing the boundary method after construction retained its incompatible
sequential-layer policy, and a real identity factor was mixed into complex
Torch gates. The final checks configure the method at construction and use
consistent complex input dtypes. Changes remain local and uncommitted.

## SU initialization and optional diagnostics

The user clarified that standalone PEPO norm/fidelity measurements should be
optional and confirmed a request for **simple-update initialization of ALS**.
Default two-site FU already skips independent pre/post measurements and
boundary convergence probes. New guards reject any unexpected standalone
norm, normalization or fidelity call during default and SU-initialized PEPO
updates with direct/DMRG boundaries. The required local norm environment,
residual and local fidelity calculation remain part of the pair solver.
Documentation now distinguishes these from optional global diagnostics.

Implemented `run(warmstart_method="simple-update")` for native PEPS/PEPO
two-site FU. Reuse the existing simple-gauge and `gate_simple` operations on
a private copy with `renorm=False`, absorb the pre-gate gauges into an
equivalent source, and project the SU guess into that source's QR/LQ bases.
The exact gated target is never truncated. The source's representation can
change outside the pair; ordinary boundary/strip version checks invalidate
old environments. This opt-in currently includes a fresh SU gauge pass per
pair, so no performance improvement is claimed. The previous reduced-SVD
initialization remains the default. Boundary `fit_init_strategy` remains a
separate control.

The shared dense ALS driver previously restarted from its own target SVD,
discarding the initial factors selected by full update. It now accepts
validated `initial_factors` for Quimb, QR and normal ALS; caller arrays are
copied and the supplied rank is retained up to `max_bond`. Native symmetry
and autodiff retain their original default initializer. FU passes its chosen
SVD/environment-gauged/SU factors through, so the initializer actually reaches
ALS. Reduced/full pair L-BFGS also consumes the SU guess. No new gate, QR,
SU or boundary kernel, dependency change, or global linear-algebra policy
was introduced. Reused the upstream/version audit above and inspected the
installed simple-gauging/absorption interfaces; classification: **adopt**.

Current validation:

- **295 focused tests passed**, `/tmp/pepo-su-focused.log`, covering native
  PEPO, PEPS full update, pair L-BFGS, shared reduced ALS, and strip batches.
  This includes **143 native PEPO module cases**, including 32 actual
  Torch CUDA/CuPy cases across SVD/SU starts, upper/lower legs, both boundary
  modes and complex64/128. Independent dense comparisons check the SU guess,
  exact target, initial solver cost and final Frobenius fidelity.
- SU tests cover reduced QR/Quimb ALS, environment gauges, reduced/full
  L-BFGS, PEPS and PEPO, custom unequal physical legs, reversed pairs,
  network exponent and complex gate scale, explicit target normalization,
  strip/layer refinement and cached/uncached replay. Initializer guards
  verify that ALS does not restart from SVD and rejects invalid shapes,
  rank, dtype, nonfinite data and unsupported routes.
- **308 integration tests passed**, `/tmp/pepo-su-integration.log`, covering
  PEPS gauges, strip/layer refinement, CuPy FU, general optimizer, full PEPO
  steps, exact warm starts and full pair objectives. These paths are separate
  from the 295-test selection above.
- **112 smoke tests passed**, `/tmp/pepo-su-smoke.log`.
- Full extended suite collected 9,719 tests but **stopped after 212 passed,
  2 skipped and 1 failed**: `test_native_reduced_loop_compression_uses_graded_svd_adapter`
  in `test_bp_symmray.py`, `/tmp/pepo-su-full.log`. The error is reduced
  tensor shape `(1, 1, 2, 2)` versus `(1, 2, 2, 2)`. It reproduces in isolation
  and after restoring the pre-change `solve_reduced_als` function from HEAD
  in a separate process: `/tmp/pepo-su-symmray-isolated.log` and
  `/tmp/pepo-su-symmray-baseline.log`. This is an existing native Symmray
  failure; no unrelated repair was made. Full-suite success is not claimed.
- `ruff check src tests`, local Markdown file-target checks and
  `git diff --check` pass. Existing upstream warnings remain, including
  expected custom-index-format warnings in the explicit metadata tests.

Native symmetry, JAX, differentiation through FU and periodic wrap bonds
remain outside this dense two-site implementation. PEPS-only layered-target
and strip-batching restrictions are unchanged; support for native PEPO FU
does not imply that every PEPS driver mode accepts a PEPO. Production-scale
SU-initialized performance was not measured. Changes remain local and
uncommitted on `develop`; no production optimization job was changed.
