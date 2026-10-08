# 2026-10-07 — Align norm reality with contraction accuracy

- Scope: fix the adaptive check that rejected tiny imaginary norm residuals
  despite meeting the requested norm/overlap accuracy; clarify whether chi was
  selected by convergence or merely reached its ceiling.
- Baselines: Pepsy develop 0ba2a05 and examples main 1a96925, with prior adaptive
  changes uncommitted. This correction also remains uncommitted and unpushed.

## Finding and fix

The policy-6 D=4 auto run used chi (128,128) as an unconverged ceiling fallback,
not as a passed selection. Its recent norm/overlap changes passed rtol=1e-5,
but a target-norm relative imaginary residual of about 1.7e-9 failed a separate
1e-12 dtype-only check. This imposed machine precision on an approximate
contraction despite a much looser requested accuracy.

Norm reality now requires `abs(Im(N))/abs(N) <= max(rtol, dtype_roundoff)`
for both measured norms. Raw complex values remain saved. Positivity,
finiteness, and fidelity bounded by `1 + dtype_roundoff` remain independent
guards; two stable cap increases, x/y agreement and fresh confirmation are
unchanged. Detailed `validity` records expose each residual, threshold and
rejection reason. `selection_status` distinguishes `converged` from
`limit_unconverged`, with a separate `at_ceiling` flag. Example execution
policy is 7 so revised and old saved runs cannot be silently conflated.

## Validation

- Focused adaptive suite: 32 passed, including six new residual/guard cases,
  real Torch and exact-reference contractions, explicit selection status, and
  strict JSON diagnostics. No need to rerun the unchanged fixed-cap suites;
  their earlier 284-pass result belongs to the D-squared implementation.
- Replayed saved production scalar probes without modifying any simulation:
  GPU0 D=4 at t=1.0 passes at chi 128 with target imaginary residual 3.94e-8;
  GPU2 D=2 at t=1.4 passes at chi 32 with residuals around 5.5e-12. Both had
  previously been flagged unconverged. Replays enforce the original retained
  cap and use the saved fresh confirmation. They do not prove a smaller cap
  would pass, nor are they new trajectory evolutions.
- Replay evidence: `/tmp/pepsy_norm_reality_replay.json`; focused test log:
  `/tmp/pepsy_norm_reality_tests.log`. Documentation and changelog updated.
- Runner/identity selection: 13 passed, 27 deselected. All three replacement
  command dry runs passed. Diff whitespace checks passed in both repositories.
- Existing dependency audit and environment are unchanged; this is a scalar
  policy correction, with no contraction/backend/library changes. Ruff remains
  unavailable in the environment.

## Running jobs

Policy-6 children originally kept running with their already-loaded old check;
Python does not pick up this source correction automatically. Replacement
policy-7 launch commands passed dry runs in separate directories, preserving
old results. A new restart would begin at t=0 and lose unfinished progress;
do not treat this correction as retroactively changing saved convergence flags.
Pending replacement roots: `/tmp/pepsy_normtol_pending_launch_roots.json`.
The user subsequently approved the restart with "yes". Parents
694180/694194/694208 and children 694575/694576/694581 exited after SIGTERM;
their outputs remain preserved. Policy-7 replacements started with parents
935190 (GPU0 D=4 auto), 935204 (GPU1 D=4 gate-by-gate), and 935218
(GPU2 D=2 auto). Active roots are in `/tmp/pepsy_adaptive_launch_roots.json`;
old roots are in `/tmp/pepsy_adaptive_launch_roots_policy6.json`. Each new
root contains exact launch commands and source snapshots. GPU3 MPS PID 89592
was not stopped or modified.
