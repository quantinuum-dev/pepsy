# 2026-09-28 — Warning, lint, and optional-configuration review

Baseline: `develop` at `5ae50de`, which commits the preceding readability and
diagnostics batch. This review follows the
[previous scoped Pylint audit](../../../history/2026-09-28-pylint-domain-fixes.md).

## Warning causes

The baseline full suite reported **5,169 passed, 129 skipped, 790 warnings**.
Its warning summary contains 92 source/message groups. Counts below are
occurrences in that run, not distinct defects:

| Cause | Baseline occurrences | Decision |
| --- | ---: | --- |
| Quimb optimizer assigns NumPy array `.shape` | 605 | Defer upstream; NumPy 2.5 deprecates this operation. No site-package edits or suppression. |
| Quimb `mode`/`method` keyword changes | 70 | Adopt installed public spelling through existing signature-based compatibility helpers. Includes 65 cold-sweep calls, one MPS trajectory call, and four test reference calls. |
| Routine Hamiltonian tests call deprecated builders | 29 | Use `ham_tn.to_mpo` / `to_pepo`; retain the explicit alias-warning regression. `Fermion.build_mpo` remains canonical and is unchanged. |
| Read-only storage, gradient scalar assertion, open figures | 3 | Copy read-only NumPy storage before Torch conversion; detach comparison scalars; close test figures after each test. |
| Compatibility-path deprecations | 24 | Retain visibility while exercising supported legacy behavior. |
| Rank-deficient Torch QR | 13 | Keep numerical warnings; a passing reconstruction does not make its ordinary derivative well-conditioned. |
| Explicit finite-value validation | 9 | Keep the warning about opt-in scans and synchronization. |
| SDCR cutoff-policy differences | 5 | Keep the warning; do not silently change truncation policy. |
| Truncation diagnostics or approximate expectations | 11 | Keep diagnostic cost and approximation warnings. |
| Tree dtype conversions or product-state layout rebuilding | 10 | Keep visibility of conversions and exact structural rebuilding. |
| Complex-to-real casts | 5 | Investigate separately; see below. |
| Other upstream contraction and NetKet notices | 6 | Four uncompressed-tree efficiency warnings, one unspecified-split-default notice, one fixed-particle-number operator performance suggestion. Retain. |
| **Total** | **790** | |

The new full run reported **5,171 passed, 129 skipped, 687 warnings in
523.06s**. All 70 keyword and 29 routine-builder warnings disappeared, as
did the read-only-storage and open-figure warnings. Quimb shape-assignment
occurrences varied from 605 to 601; that upstream issue is not fixed.
Torch's once-per-process scalar warning moved to a second qMERA assertion;
that assertion was subsequently detached and the qMERA module rechecked
with this warning promoted to an error. The full count above precedes that
last test-only edit.

Two new Loky worker-stop notices appeared during the NumPy exact-batch
phase-pass test. The test passed, but the notices can indicate worker timeout
or process resource problems; this run does not establish their cause.
An isolated rerun is recorded in the validation section. No executor or
warning policy was changed to conceal them.

The five cast warnings are not all certified harmless:

- Two originate in Quimb BP message initialization (`bp_common.py`). Inspect
  imaginary components and backend dtype policy before changing the messages.
- One comes from the real-dtype `hrs_to_peps` fixture in
  `test_gate_simple_2d_routes_swaps_with_real_torch_dtype`: a complex Haar
  vector is cast to real. The SWAP test passes, but that does not establish
  an appropriate real-state sampling distribution.
- Two arise in `test_real_coefficient_state_keeps_y_rotation`, one in
  stabilizer constant construction and one in Autoray's Torch conversion.
  The rotation regression passes. Any cleanup must preserve the real
  coefficient representation of Y rotations and test its phases explicitly.

No warning suppression was added. Regression checks promote the targeted
warnings to errors. Figure teardown releases already-imported
Matplotlib figures without importing Matplotlib for other tests.

## Standard Pylint triage

Repeated the previous command with `--rcfile=/dev/null --persistent=n` over
`bp/`, `vmc/torch/`, `optimizers/mpo/`, and operator `mpo*.py` / `pepo*.py`:

| Finding class | Before | After |
| --- | ---: | ---: |
| Total | 1,802 | 1,795 |
| Error-labelled | 71 | 71 |
| Warning | 298 | 297 |
| Convention | 348 | 340 |
| Refactor | 1,085 | 1,087 |
| Broad exception catch | 32 | 31 |
| Missing function docstring | 39 | 31 |

Exit status remains **30**. This is a triage result, not clean Pylint.
Line-based duplicate-code reports shift when docstrings change; total count
alone is not a design-quality metric. No lint suppressions were introduced.
The prior audit records runtime checks for the remaining error-labelled
dynamic exports, inherited methods, and optional imports.

### Exception decisions

- **Fixed:** reduced-update `_metric_weight_factor` catches NumPy numerical
  factorization failures and backend `RuntimeError`, allowing the existing
  PSD eigendecomposition fallback. Unrelated `TypeError` now propagates.
  A focused regression checks both fallback exception families and propagation.
- **Retain:** MPO transactional handlers restore state and record failure,
  then re-raise or use the explicitly requested recovery policy. Narrowing
  them could leave state partially updated. Overlap diagnostics retain an
  error string without invalidating an otherwise successful fit.
- **Retain:** backend inference, operator fingerprint/equality, symbolic
  rank selection, and Symmray capability probes have conservative fallback
  policies. Failure means less deduplication, static rank, or an unavailable
  capability, rather than a fabricated numerical result.
- **Defer:** BP copy/ones adapters need a backend-specific failure matrix
  before narrowing their fallback catches; avoid introducing dense conversion
  for native graded arrays.
- **Prioritize follow-up:** Torch boundary reuse has catches that select
  another axis or full-amplitude evaluation. Compilation reports already
  retain failures, but some runtime reuse catches discard the cause. Add
  bounded, opt-in fallback diagnostics before narrowing them. Preserve
  explicit approximation policy and propagate failures of the final route.

### Ranked large-function follow-up

These are reviewed extraction candidates, not changes made in this pass.
AST line spans include signatures, docstrings, and nested helpers:

| Priority | Owner / candidate | Concrete next improvement and invariant |
| --- | --- | --- |
| 1 | Torch boundary `connected_amplitudes` (459 lines) | Separate reuse-job preparation, evaluation, and result/statistics assembly. Preserve target row order, diagonal reuse, autograd lifetime, and full-amplitude fallback. |
| 2 | BP `_contract_open_scalar_edges` (396) and `compute_local_expectation_open_loop_series` (429) | Separate enumeration/budget handling from numerator/denominator evaluation. Preserve native fermion signs, cache keys, and exact/corridor policy. |
| 3 | Torch `estimate_observables` (412) | Share result assembly across fixed-sample and legacy-sweep paths without changing ESS/R-hat stopping rules or sample reuse. |
| 4 | PEPO `_add_generic_cluster_levels` (327) | Separate shape/orbit preparation from residual factorization and block insertion. Each order must subtract the complete lower-order PEPO; retain physical ordering and rank policy. |
| 5 | MPO `_run_dmrg` (351) | Keep existing target/guess helpers; consider isolating stream accounting next. Preserve channel events, transaction rollback, counters, and sampled diagnostics. |

BP `compress_bond_cluster` is 418 lines but includes an extensive numerical
contract and explicit input checks. Do not split it solely to reduce the
line count. Eight missing docstrings were filled in for backend helper
semantics and the PNE message accessor. Other missing property/helper
docstrings remain; public shape and ownership contracts took priority.

## Optional configurations

The original 129 skips group as 58 missing CuPy, 43 unavailable CUDA,
25 MPI rank requirements, two JAX device requirements, and one Metal check.
Additional runs in this session established:

| Configuration | Result | Limit |
| --- | --- | --- |
| MPI, two ranks | 25 passed per rank | Open MPI 5.0.10, local processes |
| MPI, three ranks | 25 passed per rank | Local processes; no multi-node claim |
| JAX, two logical CPU devices | 2 passed | Non-default-device placement, not GPU execution |
| Torch Metal probe plus CPU controls | 4 passed, 1 skipped | Metal scalar ledger passed; native Metal QR is unsupported by this Torch build |
| CUDA / CuPy | Unavailable | No NVIDIA GPU; CuPy not installed |

Metal was unavailable inside the execution sandbox but available in the
approved host probe. The Metal QR check skipped for missing
`aten::linalg_qr.out`; CPU fallback was not enabled. MPI also required host
execution for its local sockets. Reproduction commands are in
[CONTRIBUTING.md](../../../CONTRIBUTING.md#choose-a-test-scope).
These targeted runs supplement the default single-process full suite;
they do not change that suite's skip conditions or prove hosted CI success.

## Upstream and API evidence

Installed: NumPy 2.5.2, Torch 2.9.1, JAX 0.8.2;
Quimb `1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`,
Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev8+gc45f91457`.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray source](https://github.com/jcmgray/symmray).
The Symmray array documentation page was unavailable during the audit;
installed implementations and official source supplied the local evidence.

**Adopt:** inspected installed environment and local-expectation signatures,
then reused Pepsy's existing keyword-capability helpers. No new backend shim,
dependency, documentation builder, or numerical approximation was added.
**Defer:** upstream Quimb's NumPy shape assignment and cast-policy questions.

Expanded [BP expectations](../../api/bp.md#expectation-inputs-and-outputs),
[MPO axes](../../api/operators/automaton.md#tensor-axes-and-ownership), and
[PEPO ownership/order](../../api/operators/cluster_expansion.md#shapes-tensor-ordering-and-ownership).
The new BP example, normalized-pair return, and input preservation passed.
Separate nonsymmetric-matrix probes confirmed MPO physical axes and PEPO's
physical transpose, dense operator orientation, and unchanged source blocks.

## Validation

- Backend / PEPS / trajectory focused checks: **143 passed, 7 skipped**.
- BP reduced-update / Hamiltonian / MPO checks: **257 passed**.
- Direct Quimb keyword reference checks: **3 passed**; upstream shape and
  numerical QR warnings remain visible.
- Ruff and the two CI mypy targets passed.
- Full suite: **5,171 passed, 129 skipped, 687 warnings in 523.06s**.
- Default smoke: **89 passed, 2 compatibility warnings in 19.62s**.
- Final qMERA module with scalar-conversion warnings promoted to errors:
  **103 passed, no warnings, in 26.70s**.
- Isolated NumPy exact-batch phase-pass test: **1 passed, no warnings,
  in 1.59s**. The full-run worker notices were not reproduced; their cause
  remains unconfirmed.
