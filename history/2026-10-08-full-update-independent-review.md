# 2026-10-08 — Independent full-update and PEPS additions review

- Scope: review the full-update implementation and recent PepsOptimizer additions.
- Branch / baseline: `develop`, `fe11e24`; reviewed changes since `e8a3f96`.
- Status: review only; no implementation fixes, staging, commits, or publication.
  This handoff is an uncommitted documentation addition.
- Existing local edits in backend config/JAX and MPS norm/compression/optimizer
  files were preserved. Tests used the current working tree.

## Findings

### P1 — An explicit FIT cap defeats adaptive chi convergence

`_calibrate_sweep_boundaries` retains `fit_max_bond` in its shared options.
Increasing the probe's `chi` therefore does not increase the actual direct
compression cap: `boundary.metrics.contract_boundary` gives an explicit
`fit_max_bond` precedence. Both scalar convergence and fresh confirmation
can compare the same restricted approximation repeatedly.

Reproduction: construct a 3x3 D2 GHZ PEPS by zeroing every site tensor and
setting its all-zero and all-one entries to one, then normalize exactly.
Configure `chi=2`, `fit_mode='direct'`, `fit_max_bond=1`,
`contraction_opt='greedy'`, and default adaptive convergence. Calibrating this
state against a copy reports `converged=True` at `(12, 12)` after probes
`(4, 4)`, `(8, 8)`, `(12, 12)`. Every actual boundary bond remains one.
The returned squared norm is 0.5, while exact contraction gives 1.0.

Reject incompatible fixed FIT caps during adaptive calibration, or make the
probe's effective FIT cap follow the selected norm/overlap cap. Ensure the
actual fitting and normalization paths use the same policy.

### P1 — Fixed-cap full update crashes with default normalization settings

`_normalize_cached_pair` passes its `chi` directly to `boundary_norm`, whose
public scalar-norm API rejects tuples. With `boundary_convergence=False`,
the resolved default normalization cap is `(4*D, 5*D)` and never gets reduced
to its norm component before this call.

A normalized dense Torch complex128 2x2 D2 PEPS, one nearest-neighbor RZZ
gate, `mode='full-update'`, and `boundary_convergence=False` raises
`TypeError: chi must be an integer when provided` after ALS with default
`normalize_final=True`. This also affects explicitly paired normalization
caps. A scalar `normalize_chi=16` avoids this error. Resolve the first component
before calling the scalar norm API, preserving supported `None` semantics.

### P2 — Clipped prechecks can reject valid full-update candidates

`_run_full_update` decides reliability from the bounded return value of
`estimate_infidelity`, instead of its retained raw value. Unlike the ordinary
sweep driver, it treats a small negative approximation clipped to zero as a
reliable perfect fit.

Controlled metric reproduction with real pair construction and ALS: return
raw pre-infidelity `-5e-4`, within the default `evaluation_negative_tol`, then
post-infidelity `0.01`. The precheck is clipped to zero and the driver records
`optimizer_rejected`. Every nonnegative postcheck loses to this invalid
baseline. Preserve and check the raw pre-estimate before postmeasurement
overwrites it; a clipped contraction artifact must not reject the ALS result.

### P2 — Full-update pre/post checks can use different caps after a retry

The fixed-cap precheck can retry at a larger cap, but `_run_full_update`
ignores `_last_evaluation_chi`. It forces the postcheck back to the original
`evaluation_chi` with retries disabled and reports that original cap as
`effective_evaluation_chi`.

Controlled metric reproduction with `infidelity_kwargs={'norm_target': None}`:
return raw values `-0.1`, `0.02`, and `0.01`. Calls use caps `(8, 10)`, `20`,
and `(8, 10)` respectively; acceptance compares the cap-20 precheck against
the cap-(8,10) postcheck. Carry the effective precheck cap into the postcheck
and step record, as the row/column driver already does.

### P2 — Full update silently ignores target normalization

`run` resolves `normalize_target` (including its `non_unitary` default), but
does not pass it to `_run_full_update`. The reduced target remains unscaled.

Reproduction: normalized 2x2 product PEPS, two-site gate `2*I`, retained cap
2, `non_unitary=True`, `normalize_target=True`, `normalize_final=False`, and
independent measurements/acceptance disabled. The output norm is 2.0 rather
than the normalized target's norm 1.0, despite exact representation being
available. Apply normalization consistently to the reduced objective and its
reconstructed target, or explicitly reject unsupported target-normalization
requests rather than accepting a no-op option.

## Validation

Environment: `~/envs/py312`, CPU-only test processes, one BLAS/OpenMP/Numba
thread and `LOKY_MAX_CPU_COUNT=2`, local `src` on `PYTHONPATH`.

- PEPS full update, boundary convergence, environment reuse, gate order,
  sweep safeguards, batching, optimizer, boundary numerics, timing, and
  sweep performance: **350 passed**, 52 warnings, 126.86 seconds.
- Shared BP reduced-update suite: **34 passed**, 14 warnings, 14.51 seconds.
- Targeted reproductions established the five findings above; the fidelity
  clipping and retry cases inject scalar metric results to test control flow.
  The normalization crash, target-scaling discrepancy, and GHZ convergence
  failure use actual tensor contractions.
- `git diff --check` passed. No implementation edits, so no new lint run.
- No complete repository suite or CUDA validation was run for this review.

Temporary reproduction script: `/tmp/pepsy_review_probes.py`; output:
`/tmp/pepsy_review_probes.log`. Test logs:
`/tmp/pepsy_review_20261008_tests.log` and
`/tmp/pepsy_review_shared_als.log`. These temporary files may disappear;
the inputs and observed failures are recorded above.

## Dependency and upstream audit

Installed versions: Torch 2.6.0+cu124, Quimb 1.15.1.dev90+g6a3906cbe,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Autoray 0.11.1.dev14+g014a3f69a, Symmray 0.4.1.dev15+g0374aaa3c.
Inspected installed `tensor_network_fit_als` and `tensor_split` signatures;
the ALS interface accepts the explicit overlap networks used here.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The Symmray abelian-array documentation URL was unavailable.
The implementation cites [Lubasch et al.](https://arxiv.org/abs/1405.3259).
Classification: **defer** dependency changes; the reproduced findings concern
Pepsy policy/control flow and do not require an upstream compatibility shim.
