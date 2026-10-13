# 2026-10-13 — Bounded PEPS boundary growth and API cleanup

- Scope: implement the agreed failure-triggered chi policy, organize its API
  and affected code, and remove duplicated cap/options handling.
- Branch / baseline: `develop` / `43dfc4a`.
- Status: working-tree changes only; no staging, commit, push or new simulation.
- The proposed whole-step TFIM PEPO target remains a follow-up by user request.

## Result

Default strip ALS now keeps its initial norm/overlap caps until an existing
diagnostic fails. `boundary_convergence="on_failure"` selects the same policy
explicitly; a mapping with `policy="on_failure"` and `max_chi` sets separate
ceilings. No routine cap probes, downward searches or SVD-tail estimators
were introduced. Each affected component doubles exactly; a doubling above
the ceiling raises `BoundaryChiLimitError` with its failed request report.

The controller is scalar-only. It reports norm-specific failures separately
from explicit nonfinite overlaps. An invalid fidelity ratio with otherwise
usable components is reported as ambiguous and requests both caps. This is
failure recovery, not convergence certification for admissible estimates.

Replay restores failed gate batches, preserving earlier accepted batches,
then reconstructs the same exact target and refits at the retained caps.
Initial normalization and final one-site normalization use the same bounded
recovery. The target norm is measured once per attempt and reused by fitting
and the postcheck. Readout comparisons use the same caps for both candidates.
Explicit exact/reused metric `chi=None` is preserved; hidden fixed FIT ranks
are rejected before mutation. Retained caps persist across replay calls.

Growth warnings and `get_boundary_growth()` include run, gate step, column
when available, stage, reason, components, old/required caps and ceiling.
See the [public policy](../docs/api/optimizers/peps.md#keep-large-caps-and-grow-only-on-failure).

## Ownership and validation

- [Shared cap controller](../src/pepsy/optimizers/_boundary_control.py) owns
  cap validation, options merging, failure signals and bounded doubling.
- [Sweep construction](../src/pepsy/optimizers/peps/_sweep_fit.py) is separated
  from gate-stream replay. Injection points and reusable contraction planning
  are preserved. Pre-fit convergence imports the shared cap validator.
- [Focused regressions](../tests/test_peps_boundary_growth.py) cover component
  attribution, exact doubling, ceiling rollback, prior accepted gates,
  retained caps, initial/target/final normalization, explicit exact metrics,
  fixed-rank conflicts, and equal healthy-path fidelity contraction counts.
- Combined PEPS driver/strip/batching/boundary-convergence/safeguard/performance
  selection: **440 passed**. The latest controller-only selection: **39 passed**.
- `python -m ruff check src tests`, `git diff --check`, and changed documentation
  file-link checks passed. These selections are not a full-suite claim.

Additional ALS/CG/L-BFGS, boundary engine/reuse, strip/layer refinement and
global-batch checks passed (**283 tests**) before three old timing fixtures
failed. The timing helper supplied NLopt's `maxeval` without selecting NLopt,
so the already-existing strip ALS defaults rejected it or bypassed the
expected legacy target-construction timing phase. The latter failure was
reproduced using the unmodified `43dfc4a` source and test in an isolated `/tmp`
archive. Set `optimizer='nlopt'` explicitly in that fixture, preserving its
intended assertions. The complete timing, public API, package-layout and
controller selection then passed: **98 tests**, including all four timing
tests and all 39 controller tests. A scalar guard also rejects empty/unknown
failure components, preventing retries that cannot increase any cap.

The legacy direct-compression precheck comparison now ignores the FIT guess
strategy, which direct compression does not consult. Existing precheck reuse
tests validate matching caps and metric policy; custom metrics still do not reuse.

## Environment and limits

Used the existing Python 3.12 environment. Installed versions were inspected:
Quimb `1.15.1.dev91+gb3bf45a50`, Autoray `0.11.1.dev14+g014a3f69a`, Cotengra
`0.8.3.dev8+g8954240f2`, Cotengrust `0.2.1`, Symmray `0.4.1.dev18+g44fc5ae53`,
Torch `2.14.1+cu126`. Public PEPS metric, optimizer, and Quimb boundary
signatures were inspected. No dependency or installed-library changes.

Reviewed [Quimb release notes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[release notes](https://cotengra.readthedocs.io/en/latest/changelog.html).
The Symmray array documentation endpoint was unavailable; its
[official repository](https://github.com/jcmgray/symmray) was used as fallback.
Upstream algorithm adoption is **deferred**: this change preserves existing
boundary/FIT kernels, array semantics and exact target construction.

No long-time D=3/4 trajectory or new accelerator trajectory is certified by
these checks. The PEPS simulation service has `MainPID=0` and no matching
benchmark Python process remains. Logs/checkpoints from the stopped t=0.4
rerun and the earlier norm-component diagnosis are preserved.
