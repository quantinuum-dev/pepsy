# 2026-10-06 — Automatic PEPS batches and review fixes

- Scope: implement automatic/fixed gate batching, explicit local NLopt
  settings, and fixes from the preceding PepsOptimizer review.
- Baseline: `develop`, `8f7c896`. Implementation, tests, docs and this evidence
  remain uncommitted working-tree edits; nothing staged or published.
- Public behavior: [PEPS optimizer guide](../../api/optimizers/peps.md).
- Prior findings: [default review](2026-10-06-peps-optimizer-default-review.md).

## Implemented

`run(k_2q_batch="auto")` is the new default. It collects an ordered stream of
two-site gates with disjoint physical endpoints, stopping before a repeated
site or a candidate whose largest target bond exceeds `2*chi`. Coordinate and
physical-index aliases resolve to the same tensor sites. Intermediate and
trailing one-site gates remain in their original circuit order. Leading
one-site gates are handled directly.

The actual routed target is checked: disjoint endpoints can share routing
bonds. A rejected candidate is discarded without consuming its queue entry.
Positive integer sizes retain count-based batching, including overlapping
gates; `1` requests the previous behavior.

A single general two-qubit gate can require up to `4*chi`; a routed target may
grow multiple bonds. An exact first gate above the budget is processed alone,
with a recorded `single_gate_exceeds_limit` reason. There is no target
truncation to enforce the batching threshold. A trial target is allocated
before checking dimensions, so this threshold is not a peak-memory limit.
Retained state compression still uses the requested `chi`.

Local NLopt defaults are explicit and overridable:
`algorithm="LD_LBFGS"`, `maxeval=100`,
`ftol_rel=ftol_abs=xtol_rel=1e-9`, `restore_best=True`.
These match existing inherited numerical defaults and apply per slice solve.
Partial per-run solver mappings preserve other constructor overrides.

Review fixes:

- Exact targets disable inherited `bond_dim` and reject explicitly truncating
  `target_gate_kwargs` values, covering the alias missed by the old guard.
- Physical-index gates remove coordinate-routing options before Quimb splits.
- Fidelity evaluation recomputes both norms by default, and forwards the
  measured target norm to variational cleanup. Known explicit norms remain
  supported.
- Invalid finite-cap metrics can retry twice at equal, doubled norm/overlap
  caps; each retry warns and is recorded. Zero `evaluation_max_retries`
  enforces strict caps. Persistent invalid estimates, nonfinite values, and
  inconsistent explicitly supplied norms remain errors. Negative roundoff is
  cleaned only within the dtype cutoff scale.
- Pre/post acceptance uses a common effective cap. If the postcheck enlarges
  it, the saved warm start is remeasured there without further retries.
- The FIT diagnostics collector retains the sweep's sequence of records.
- `run(mode=...)` is temporary; `set_mode` changes the configured mode.
- API documentation now correctly identifies NLopt as the default for
  Torch-backed Symmray as well as dense states.

## Fresh validation

Activated the shared Python 3.12 environment. Numerical runs used
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

```text
python -m pytest -q -ra -o addopts='' \
  tests/test_peps_optimizer_batching.py tests/test_optimize_peps.py \
  tests/test_optimize_global.py tests/test_prepare_boundary_inputs.py \
  tests/test_public_api.py tests/test_package_layout.py
```

**402 passed**, 15 warnings, no skips, 25.53 seconds on the final implementation.
Ruff (`python -m ruff check src tests`) and `git diff --check` passed.

New regressions cover dense circuit reconstruction with intervening one-site
gates; integer sizes 1/2/3; shared-site aliases on Torch; singleton overflow;
crossing routed gates and rollback; the `bond_dim` guard; bounded metric
retries; common-cap acceptance; target-norm forwarding; mode restoration;
FIT record collection; NLopt defaults and partial overrides; and native
Torch U1 fermionic batching.

The recorded 4x4 identity failure now retries successfully and agrees with
the input state up to normalization at dense fidelity one. In the native
2x2 U1 check, two disjoint hopping gates form one batch and agree with separate
application at fidelity one, preserving U1FermionicArray storage and Torch
CPU complex128 blocks. Existing selected suites also exercise native U1/U1U1
sweeps and complex64/complex128 metric/compression paths.

During development, one old test expected the former implicit unit norms;
it now checks norm recomputation. An initial singleton-overflow test used a
2x2 network whose physical rank ceiling fits `2*chi`; it was replaced with a
3x3 interior-edge fixture that actually exceeds the budget while retaining
the dense exact-target assertion. Neither failure was hidden by loosening
numerical tolerances. Earlier 329/401-test results are superseded for this
implementation by the final selection above.

No full package suite, GPU validation, broad PEPO/cyclic matrix, or throughput
benchmark was run. Automatic batching changes where truncations occur and
therefore may change approximate results relative to `k_2q_batch=1`.
Recomputed norms and retries cost extra contractions; the default caps are
starting accuracy settings, not a guarantee of convergence or a hard retry
memory budget.

## Environment and upstream evidence

Reused the same session's installed-version/signature audit and official
upstream checks recorded in the [review](2026-10-06-peps-optimizer-default-review.md#upstream-audit).
The environment and dependencies were not modified. Native gates use the
existing public Fermion/SymPEPS interfaces and graded Symmray contractions.
Classification: **adopt** existing public gate/compression APIs; **defer**
dependency upgrades or new upstream shims. No library internals were vendored
or edited.

Concurrent gradient-solver backend-validation edits appeared in
`src/pepsy/solvers/gradient.py`, its API guide, JAX tests, and a separate
changelog bullet during this work. They were preserved and are not part of
this task's implementation. The final checks ran in that shared working tree.
