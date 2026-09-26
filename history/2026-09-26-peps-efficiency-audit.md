# 2026-09-26 — PEPS correctness and efficiency audit

- Scope: user-requested careful review of every PEPS stage against Quimb and
  Pepsy, with efficiency fixes and fresh numerical validation.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: working-tree edits only, unstaged and uncommitted. Existing
  unrelated changes and production jobs preserved; no production PEPS launch.

## Findings and changes

- Rechecked public/installed gate, gauge, boundary, and BP implementations.
  Reused the unchanged dependency audit from the implementation session and
  reviewed the linked Quimb example and latest 2D/changelog documentation.
- The example equilibrates gauges after each step; the first engine only
  initialized them and updated the gated bonds. Added explicit refresh
  frequency/tolerance/iteration controls, defaulting to every actual step,
  1e-6, and 1000 iterations. Equilibration uses zero extra truncation and
  Quimb's physical-scale bookkeeping. Save its residual, iterations, and
  freshness status. This changes the default subsequent SU truncation policy;
  `--peps-gauge-every 0` selects initial-only gauging. Sweep reuse checks include
  these controls. Gate ordering remains the shared second-order Strang stream.
- Cache up to four exact interval/Hamiltonian gate streams, use the bundled
  public gate API with explicit dense reduce-split, and cache local operators.
  Transfer local Z scalars to the host together.
- Expand loops on the BP object already solved for the snapshot, avoiding
  reconstruction of D2BP's contraction expressions. Reuse only fixed loop
  geometry across times, never numeric contractions or BP fixed points.
- A 3x4 finite-chi readout exposed an overly strict imaginary-part guard.
  Approximate boundary environments can break Hermiticity. Report real
  Hermitian expectations with their imaginary residuals and explicit
  `check_boundary_chi` status; retain failures for nonfinite outputs or a
  nonpositive real global norm. These status flags are not error bounds.
- Added opt-in stage durations/call counts through `--timing`, with explicit
  GPU synchronization through `--timing-sync-device`. No timing barriers by
  default. Updated runner documentation and benchmark AGENTS guidance.
- A potentially faster layer-by-layer global norm route was evaluated but
  deferred because it changed finite-chi errors. No package implementation,
  dependencies, contraction policy, or cutoff was silently replaced.

## New numerical evidence

- Gauge equilibration preserved a scaled 3x4 state to 1.34e-14 relative error.
- Direct BP reuse matched the reconstructing wrapper to 7e-16 for four-site
  regions and roundoff for six-site regions; scaled-state tests guard norm
  versus norm-squared conventions and stale contraction caches.
- On a 3x4 evolved PEPS with D=4, five dt=0.2 steps, boundary local-Z errors
  versus dense contraction were 1.26e-4 (chi=8), 2.49e-6 (chi=16), and
  3.25e-15 (chi=32). Imaginary residuals decreased correspondingly. This is
  measurement convergence for the represented PEPS, not proof of evolution
  accuracy at D=4.
- Tiny 2x3 CUDA tests with gauge equilibration and clipped intervals preserved
  dtype/device and matched dense evolution: errors 4.71e-15 (complex128) and
  4.80e-7 (complex64). Both boundary and full-region loop norms agreed with
  dense contraction. The complex64 probe used gauge/BP tolerances of 1e-5.
- Existing Pepsy simple-update suite: **10 passed**. Focused PEPS suite before
  the final finite-chi regression: **14 passed**. Final full benchmark suite:
  **399 passed, 65 warnings**, 104.04 s, including all 15 PEPS cases. Ruff for
  Pepsy `src tests` and benchmark implementation/tests, documentation links,
  and both repositories' whitespace checks passed. No full Pepsy suite claimed.
- Bounded CPU stage timings and the full algorithm/audit notes are retained in
  the [numerical note](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/development/notes/peps_simple_update.md).
  Temporary artifacts: `/tmp/peps-audit-before.prof`,
  `/tmp/peps-audit-profile.json`, `/tmp/peps-audit-full.log`.

## Remaining scope

Large 5x6 PEPS performance and convergence remain unmeasured. The existing
OBC/dense NumPy/Torch scope remains; no shot sampling, entropy, PBC, state
serialization, or PEPS disk cache was added. Gauge refresh frequency can
trade cost against later truncation behavior and is an explicit control.
The current [runner guide](../../pepsy_examples/experiments/mps_magnetization/benchmark/magnetization/README.md#peps-simple-update)
documents all controls, timings, and output fields.
